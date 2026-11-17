"""Data acquisition from ESPN's public NBA scoreboard API.

Source
------
Endpoint: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
Auth:     none (free, public, unofficial)
Format:   JSON

Query parameters we use:
  dates=YYYYMMDD   -> all games on one day
  dates=YYYYMM     -> all games in one calendar month (needs limit=500)
  limit=500        -> max events returned

Structure (trimmed to the fields we read):
  {
    "events": [
      {
        "id": "401909993",
        "date": "2026-11-16T03:00Z",                 # tip-off, UTC
        "season": {"year": 2027, "slug": "regular-season"},
        "status": {"type": {"name": "STATUS_FINAL", "completed": true}},
        "competitions": [{
          "neutralSite": false,
          "attendance": 18236,
          "venue": {"fullName": "Kia Center"},
          "competitors": [{
            "homeAway": "home",
            "score": "103",
            "winner": false,
            "team": {"abbreviation": "SAC", "displayName": "Sacramento Kings"},
            "statistics": [{"name": "rebounds", "displayValue": "37"}, ...],
            "records": [{"type": "total", "summary": "1-12"}, ...]
          }, {...away team...}]
        }]
      }
    ]
  }

Each event is flattened into ONE row per game (home_* and away_* columns).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
EASTERN = ZoneInfo("America/New_York")
TIMEOUT_SECONDS = 30

# ESPN stat name -> our column suffix
STAT_MAP = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "fieldGoalsMade": "fgm",
    "fieldGoalsAttempted": "fga",
    "fieldGoalPct": "fg_pct",
    "threePointFieldGoalsMade": "fg3m",
    "threePointFieldGoalsAttempted": "fg3a",
    "threePointFieldGoalPct": "fg3_pct",
    "freeThrowsMade": "ftm",
    "freeThrowsAttempted": "fta",
    "freeThrowPct": "ft_pct",
}


class DataFetchError(RuntimeError):
    """Raised when the ESPN API can't be reached or returns something unexpected."""


def fetch_scoreboard(dates: str, session: requests.Session | None = None) -> dict:
    """Call the ESPN scoreboard endpoint for a day (YYYYMMDD) or month (YYYYMM)."""
    http = session or requests
    try:
        resp = http.get(
            SCOREBOARD_URL,
            params={"dates": dates, "limit": 500},
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": "nba-mock-sportsbook/1.0 (student project)"},
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        raise DataFetchError(f"Could not reach ESPN API for dates={dates}: {exc}") from exc
    except ValueError as exc:  # bad JSON
        raise DataFetchError(f"ESPN API returned invalid JSON for dates={dates}") from exc

    if "events" not in payload:
        raise DataFetchError(f"Unexpected ESPN response for dates={dates}: {str(payload)[:200]}")
    return payload


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_competitor(comp: dict, prefix: str) -> dict:
    team = comp.get("team", {})
    row = {
        f"{prefix}_team": team.get("abbreviation"),
        f"{prefix}_team_name": team.get("displayName"),
        f"{prefix}_score": _to_float(comp.get("score")),
    }
    stats = {s.get("name"): s.get("displayValue") for s in comp.get("statistics", [])}
    for espn_name, col in STAT_MAP.items():
        row[f"{prefix}_{col}"] = _to_float(stats.get(espn_name))
    records = {r.get("type"): r.get("summary") for r in comp.get("records", [])}
    row[f"{prefix}_record"] = records.get("total")
    return row


def parse_events(payload: dict) -> list[dict]:
    """Flatten the raw ESPN JSON into a list of one-row-per-game dicts."""
    rows = []
    for ev in payload.get("events", []):
        try:
            comp = ev["competitions"][0]
            competitors = {c["homeAway"]: c for c in comp["competitors"]}
            home, away = competitors["home"], competitors["away"]
        except (KeyError, IndexError):
            log.warning("Skipping malformed event %s", ev.get("id"))
            continue

        status = ev.get("status", {}).get("type", {})
        tipoff_utc = pd.to_datetime(ev.get("date"), utc=True)
        completed = bool(status.get("completed"))

        row = {
            "game_id": str(ev.get("id")),
            "tipoff_utc": tipoff_utc.isoformat(),
            "game_date": tipoff_utc.tz_convert(EASTERN).date().isoformat(),
            "season": int(ev.get("season", {}).get("year", 0)),
            "season_type": ev.get("season", {}).get("slug"),
            "status": status.get("name"),
            "completed": completed,
            "neutral_site": bool(comp.get("neutralSite", False)),
            "venue": comp.get("venue", {}).get("fullName"),
            "attendance": _to_float(comp.get("attendance")) or None,
        }
        row.update(_parse_competitor(home, "home"))
        row.update(_parse_competitor(away, "away"))

        if not completed:
            # ESPN reports "0" for games not yet played; that's not a real score.
            row["home_score"] = row["away_score"] = None
        rows.append(row)
    return rows


def month_keys(start: date, end: date) -> list[str]:
    """Return YYYYMM strings for every month from start to end inclusive."""
    keys, cur = [], date(start.year, start.month, 1)
    while cur <= end:
        keys.append(cur.strftime("%Y%m"))
        cur = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
    return keys


def fetch_games_for_months(months: list[str]) -> pd.DataFrame:
    """Download and flatten every game in the given months."""
    rows: list[dict] = []
    with requests.Session() as session:
        for m in months:
            rows.extend(parse_events(fetch_scoreboard(m, session=session)))
    return pd.DataFrame(rows)


def fetch_upcoming_games(days_ahead: int = 7, today: date | None = None) -> pd.DataFrame:
    """Download the next `days_ahead` days of the schedule (day by day)."""
    today = today or datetime.now(EASTERN).date()
    rows: list[dict] = []
    with requests.Session() as session:
        for offset in range(days_ahead + 1):
            day = today + timedelta(days=offset)
            rows.extend(parse_events(fetch_scoreboard(day.strftime("%Y%m%d"), session=session)))
    return pd.DataFrame(rows)
