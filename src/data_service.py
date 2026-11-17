"""Glue between the ESPN API and the SQLite cache.

refresh_games() only downloads what it needs:
  * months that are fully in the past are fetched once and marked complete;
  * the current month (and upcoming days) are re-fetched on every refresh,
    so scores and schedules stay current.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src import database as db
from src.data_fetcher import EASTERN, fetch_games_for_months, fetch_upcoming_games, month_keys


def season_start_year(today: date) -> int:
    """NBA seasons start in October. Nov 2026 -> 2026 (the 2026-27 season)."""
    return today.year if today.month >= 8 else today.year - 1


def default_history_start(today: date) -> date:
    """Pull the previous full season plus the current one (more data for EDA/ratings)."""
    return date(season_start_year(today) - 1, 10, 1)


def refresh_games(today: date | None = None, days_ahead: int = 7, db_path=None) -> dict:
    """Fetch any missing data from ESPN into SQLite. Returns a small summary dict."""
    today = today or datetime.now(EASTERN).date()
    db.init_db(db_path)

    done = db.completed_months(db_path)
    current_month = today.strftime("%Y%m")
    wanted = [m for m in month_keys(default_history_start(today), today) if m not in done]
    # Skip the NBA off-season months (July-September) - there are no games.
    wanted = [m for m in wanted if m[-2:] not in {"07", "08", "09"}]

    n_rows = 0
    for m in wanted:
        df = fetch_games_for_months([m])
        n_rows += db.upsert_games(df, db_path)
        db.log_month_fetch(m, complete=(m < current_month), db_path=db_path)

    upcoming = fetch_upcoming_games(days_ahead=days_ahead, today=today)
    n_rows += db.upsert_games(upcoming, db_path)
    return {"months_fetched": wanted, "rows_upserted": n_rows, "refreshed_at": db.now_iso()}


def nba_games(games: pd.DataFrame) -> pd.DataFrame:
    """Drop preseason and exhibition games.

    ESPN labels the All-Star Game (e.g. team "STARS") as regular season, so we also
    drop any game involving a "team" that appears fewer than 10 times in the data -
    real NBA teams play 82 games a season.
    """
    if games.empty:
        return games
    g = games[games["season_type"].isin(["regular-season", "post-season", "play-in-season"])]
    counts = pd.concat([g["home_team"], g["away_team"]]).value_counts()
    real = set(counts[counts >= 10].index)
    return g[g["home_team"].isin(real) & g["away_team"].isin(real)].copy()


SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "games_snapshot.csv"


def load_snapshot_into_db(db_path=None) -> int:
    """Offline fallback: seed the cache from the CSV committed to the repo."""
    if not SNAPSHOT_PATH.exists():
        return 0
    db.init_db(db_path)
    return db.upsert_games(pd.read_csv(SNAPSHOT_PATH, dtype={"game_id": str}), db_path)


def export_snapshot(db_path=None) -> Path:
    db.load_games(db_path).drop(columns=["fetched_at"]).to_csv(SNAPSHOT_PATH, index=False)
    return SNAPSHOT_PATH
