"""Player data (box scores) and player-prop lines.

Source: ESPN game summary endpoint (free, no key, JSON)
    https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<game_id>
    boxscore.players[team].statistics[0] -> {keys: [...], athletes: [{athlete, starter, didNotPlay, stats}]}

Each completed game is downloaded once and stored as one row per player per game
in `player_games`. A CSV snapshot ships with the repo, so a first launch only
downloads games played since the snapshot was taken.

Props: the line for a stat is a recency-weighted average of the player's recent
games, rounded to .5. P(over) comes from a normal model using the player's own
game-to-game standard deviation.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src import database as db
from src.data_fetcher import TIMEOUT_SECONDS, DataFetchError
from src.odds_model import FAIR_PRICE, STANDARD_JUICE, normal_cdf, round_half

log = logging.getLogger(__name__)

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "player_games_snapshot.csv"
PROP_STATS = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3-pointers made"}
MIN_SD = {"pts": 4.0, "reb": 1.8, "ast": 1.5, "fg3m": 0.9}
HALF_LIFE_GAMES = 8


def _num(v) -> float | None:
    try:
        return float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None


def _made_att(v) -> tuple[float | None, float | None]:
    try:
        m, a = str(v).split("-")
        return float(m), float(a)
    except ValueError:
        return None, None


def parse_box_score(payload: dict, game: dict) -> list[dict]:
    """Flatten one ESPN summary payload into player rows."""
    rows = []
    teams = payload.get("boxscore", {}).get("players", [])
    abbrs = [t.get("team", {}).get("abbreviation") for t in teams]
    for t in teams:
        team = t.get("team", {}).get("abbreviation")
        opponent = next((a for a in abbrs if a != team), None)
        for block in t.get("statistics", [])[:1]:
            keys = block.get("keys", [])
            for a in block.get("athletes", []):
                ath = a.get("athlete", {})
                stats = dict(zip(keys, a.get("stats", []), strict=False))
                fgm, fga = _made_att(stats.get("fieldGoalsMade-fieldGoalsAttempted"))
                fg3m, fg3a = _made_att(stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
                ftm, fta = _made_att(stats.get("freeThrowsMade-freeThrowsAttempted"))
                dnp = bool(a.get("didNotPlay")) or not a.get("stats")
                rows.append({
                    "game_id": str(game["game_id"]), "player_id": str(ath.get("id")),
                    "season": int(game["season"]), "game_date": str(pd.Timestamp(game["game_date"]).date()),
                    "team": team, "opponent": opponent, "is_home": int(team == game.get("home_team")),
                    "name": ath.get("displayName"), "position": (ath.get("position") or {}).get("abbreviation"),
                    "headshot": (ath.get("headshot") or {}).get("href"),
                    "starter": int(bool(a.get("starter"))), "dnp": int(dnp), "dnp_reason": a.get("reason"),
                    "minutes": _num(stats.get("minutes")), "pts": _num(stats.get("points")),
                    "reb": _num(stats.get("rebounds")), "ast": _num(stats.get("assists")),
                    "fg3m": fg3m, "fg3a": fg3a, "fgm": fgm, "fga": fga, "ftm": ftm, "fta": fta,
                    "stl": _num(stats.get("steals")), "blk": _num(stats.get("blocks")),
                    "tov": _num(stats.get("turnovers")), "plus_minus": _num(stats.get("plusMinus")),
                })
    return rows


def fetch_box_score(game: dict, session: requests.Session) -> list[dict] | None:
    try:
        r = session.get(SUMMARY_URL, params={"event": game["game_id"]}, timeout=TIMEOUT_SECONDS)
        if r.status_code == 404:
            return None  # ESPN has no box score for this game (permanent)
        r.raise_for_status()
        return parse_box_score(r.json(), game)
    except (requests.RequestException, ValueError) as exc:
        raise DataFetchError(f"Could not fetch box score for game {game['game_id']}: {exc}") from exc


def load_snapshot_if_empty(db_path=None) -> int:
    """Seed player data from the bundled CSV on first launch."""
    if db.fetched_box_ids(db_path) or not SNAPSHOT_PATH.exists():
        return 0
    df = pd.read_csv(SNAPSHOT_PATH, dtype={"game_id": str, "player_id": str})
    return db.upsert_player_games(df, sorted(df["game_id"].unique()), db_path)


def refresh_player_logs(games: pd.DataFrame, seasons: list[int] | None = None, db_path=None,
                        max_workers: int = 8) -> int:
    """Download box scores for completed games not yet stored. Returns games fetched."""
    db.init_db(db_path)
    load_snapshot_if_empty(db_path)
    seasons = seasons or [int(games["season"].max())]
    done = db.fetched_box_ids(db_path)
    todo = games[games["completed"] & games["season"].isin(seasons) & ~games["game_id"].isin(done)]
    if todo.empty:
        return 0
    records = todo[["game_id", "season", "game_date", "home_team"]].to_dict("records")
    rows, ok_ids = [], []
    with requests.Session() as session, ThreadPoolExecutor(max_workers) as pool:
        futures = {pool.submit(fetch_box_score, g, session): g["game_id"] for g in records}
        for fut, gid in futures.items():
            try:
                game_rows = fut.result()
            except DataFetchError as exc:
                log.warning("%s", exc)
                continue
            if game_rows is None:  # permanently missing: mark as fetched so we don't retry
                ok_ids.append(gid)
            elif game_rows:  # an empty box score means ESPN hasn't posted it yet; retry next refresh
                rows.extend(game_rows)
                ok_ids.append(gid)
    db.upsert_player_games(pd.DataFrame(rows), ok_ids, db_path)
    return len(ok_ids)


def export_snapshot(db_path=None) -> Path:
    db.load_player_games(db_path).to_csv(SNAPSHOT_PATH, index=False)
    return SNAPSHOT_PATH


# ---------------------------------------------------------------- stats
def played(pg: pd.DataFrame) -> pd.DataFrame:
    return pg[~pg["dnp"] & pg["minutes"].fillna(0).gt(0)]


def season_averages(pg: pd.DataFrame, season: int, min_games: int = 1) -> pd.DataFrame:
    """Per-player season averages (games played only), latest team, headshot."""
    p = played(pg[pg["season"] == season]).sort_values("game_date")
    if p.empty:
        return pd.DataFrame()
    agg = p.groupby("player_id").agg(
        name=("name", "last"), team=("team", "last"), position=("position", "last"), headshot=("headshot", "last"),
        gp=("game_id", "nunique"), mpg=("minutes", "mean"), ppg=("pts", "mean"), rpg=("reb", "mean"),
        apg=("ast", "mean"), tpm=("fg3m", "mean"), spg=("stl", "mean"), bpg=("blk", "mean"),
        fgm=("fgm", "sum"), fga=("fga", "sum"), fg3m_tot=("fg3m", "sum"), fg3a=("fg3a", "sum"),
        ftm=("ftm", "sum"), fta=("fta", "sum"),
    )
    agg["fg_pct"] = agg["fgm"] / agg["fga"].where(agg["fga"] > 0)
    agg["fg3_pct"] = agg["fg3m_tot"] / agg["fg3a"].where(agg["fg3a"] > 0)
    agg["ft_pct"] = agg["ftm"] / agg["fta"].where(agg["fta"] > 0)
    agg = agg[agg["gp"] >= min_games].drop(columns=["fgm", "fga", "fg3m_tot", "fg3a", "ftm", "fta"])
    return agg.sort_values("ppg", ascending=False).reset_index()


# ---------------------------------------------------------------- props
def prop_line(values: pd.Series, stat: str) -> tuple[float, float, float]:
    """(line, mean, sd) from a player's recent games, most recent last."""
    v = values.dropna().to_numpy()[-20:]
    w = 0.5 ** (np.arange(len(v))[::-1] / HALF_LIFE_GAMES)
    mean = float(np.average(v, weights=w))
    sd = max(float(np.sqrt(np.average((v - mean) ** 2, weights=w))), MIN_SD[stat])
    line = round_half(mean)
    if line == round(line):  # always x.5: no pushes on props
        line += 0.5 if mean >= line else -0.5
    return max(line, 0.5), mean, sd


def prop_menu(pg: pd.DataFrame, team: str, opponent: str, house_edge: bool, n_players: int = 4,
              min_games: int = 5) -> list[dict]:
    """Props for a team's top players by recent minutes (who played the team's latest game)."""
    tg = pg[pg["team"] == team].sort_values("game_date")
    if tg.empty:
        return []
    last_game = tg["game_id"].iloc[-1]
    active = set(tg.loc[(tg["game_id"] == last_game) & ~tg["dnp"], "player_id"])
    p = played(tg[tg["player_id"].isin(active)])
    recent = p.groupby("player_id").tail(10)
    top = recent.groupby("player_id")["minutes"].mean().sort_values(ascending=False).head(n_players).index
    price = STANDARD_JUICE if house_edge else FAIR_PRICE
    menu = []
    for pid in top:
        games = p[p["player_id"] == pid]
        if len(games) < min_games:
            continue
        info = games.iloc[-1]
        for stat in PROP_STATS:
            if stat == "fg3m" and games["fg3m"].tail(10).mean() < 1.0:
                continue  # skip 3PM props for non-shooters
            line, mean, sd = prop_line(games[stat], stat)
            p_over = 1 - normal_cdf((line - mean) / sd)
            menu.append({"player_id": pid, "player_name": info["name"], "team": team, "opponent": opponent,
                         "headshot": info["headshot"], "position": info["position"], "stat": stat,
                         "line": line, "mean": mean, "sd": sd, "p_over": p_over,
                         "over_odds": price, "under_odds": price, "games": len(games)})
    return menu


def grade_prop(selection: str, line: float, value: float) -> str:
    if value == line:
        return "push"
    return "won" if (value > line) == (selection == "OVER") else "lost"
