"""SQLite storage: cached games, user wallets, and bets.

Everything lives in one local file (data/sportsbook.db) so the app needs no
database server. Games are cached so we don't hit the ESPN API on every page load.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path(os.getenv("SPORTSBOOK_DB", Path(__file__).resolve().parent.parent / "data" / "sportsbook.db"))
STARTING_BALANCE = 1000.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    tipoff_utc TEXT, game_date TEXT, season INTEGER, season_type TEXT,
    status TEXT, completed INTEGER, neutral_site INTEGER, venue TEXT, attendance REAL,
    home_team TEXT, home_team_name TEXT, home_score REAL, home_record TEXT,
    home_pts REAL, home_reb REAL, home_ast REAL, home_fgm REAL, home_fga REAL, home_fg_pct REAL,
    home_fg3m REAL, home_fg3a REAL, home_fg3_pct REAL, home_ftm REAL, home_fta REAL, home_ft_pct REAL,
    away_team TEXT, away_team_name TEXT, away_score REAL, away_record TEXT,
    away_pts REAL, away_reb REAL, away_ast REAL, away_fgm REAL, away_fga REAL, away_fg_pct REAL,
    away_fg3m REAL, away_fg3a REAL, away_fg3_pct REAL, away_ftm REAL, away_fta REAL, away_ft_pct REAL,
    fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS fetch_log (
    month TEXT PRIMARY KEY,      -- YYYYMM
    fetched_at TEXT,
    complete INTEGER             -- 1 = month is in the past, never re-fetch
);
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    balance REAL NOT NULL,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS bets (
    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL REFERENCES users(username),
    game_id TEXT NOT NULL,
    matchup TEXT,
    bet_type TEXT NOT NULL,      -- moneyline | spread | total
    selection TEXT NOT NULL,     -- team abbreviation, or OVER / UNDER
    line REAL,                   -- spread for the selected team, or the total
    american_odds INTEGER NOT NULL,
    stake REAL NOT NULL,
    potential_payout REAL NOT NULL,  -- stake + profit if it wins
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | won | lost | push | void
    payout REAL DEFAULT 0,
    placed_at TEXT,
    settled_at TEXT
);
"""

GAME_COLUMNS = [
    "game_id", "tipoff_utc", "game_date", "season", "season_type", "status", "completed",
    "neutral_site", "venue", "attendance",
    *[f"{side}_{c}" for side in ("home", "away") for c in (
        "team", "team_name", "score", "record", "pts", "reb", "ast", "fgm", "fga", "fg_pct",
        "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct")],
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path | str | None = None):
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- games cache
def upsert_games(df: pd.DataFrame, db_path=None) -> int:
    if df.empty:
        return 0
    df = df.reindex(columns=GAME_COLUMNS).copy()
    df["fetched_at"] = now_iso()
    df = df.astype(object).where(df.notna(), None)
    cols = list(df.columns)
    sql = (f"INSERT OR REPLACE INTO games ({', '.join(cols)}) "
           f"VALUES ({', '.join('?' * len(cols))})")
    with connect(db_path) as conn:
        conn.executemany(sql, df.itertuples(index=False, name=None))
    return len(df)


def load_games(db_path=None) -> pd.DataFrame:
    with connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM games", conn)
    if not df.empty:
        df["completed"] = df["completed"].astype(bool)
        df["neutral_site"] = df["neutral_site"].astype(bool)
        df["tipoff_utc"] = pd.to_datetime(df["tipoff_utc"], utc=True)
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def completed_months(db_path=None) -> set[str]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT month FROM fetch_log WHERE complete = 1").fetchall()
    return {r["month"] for r in rows}


def log_month_fetch(month: str, complete: bool, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO fetch_log VALUES (?, ?, ?)", (month, now_iso(), int(complete)))


# ---------------------------------------------------------------- users
def get_or_create_user(username: str, db_path=None) -> dict:
    username = username.strip()
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO users VALUES (?, ?, ?)", (username, STARTING_BALANCE, now_iso()))
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row)


def get_balance(username: str, db_path=None) -> float:
    with connect(db_path) as conn:
        row = conn.execute("SELECT balance FROM users WHERE username = ?", (username,)).fetchone()
    return float(row["balance"]) if row else 0.0


def reset_user(username: str, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM bets WHERE username = ?", (username,))
        conn.execute("UPDATE users SET balance = ? WHERE username = ?", (STARTING_BALANCE, username))


def leaderboard(db_path=None) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(
            """SELECT u.username, u.balance,
                      COUNT(b.bet_id) AS bets,
                      SUM(CASE WHEN b.status = 'won' THEN 1 ELSE 0 END) AS wins,
                      SUM(CASE WHEN b.status = 'pending' THEN b.stake ELSE 0 END) AS at_risk
               FROM users u LEFT JOIN bets b ON b.username = u.username
               GROUP BY u.username ORDER BY u.balance DESC""",
            conn,
        )


def house_summary(db_path=None) -> dict:
    """The house's side of every settled bet: money taken in, paid out, and kept.

    Every stake leaves the bettor's balance when the bet is placed. When a bet settles,
    the house pays back stake + winnings (won), the stake (push/void), or nothing (lost).
    So house profit = total stakes - total payouts.
    """
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS bets, COALESCE(SUM(stake), 0) AS handle, COALESCE(SUM(payout), 0) AS paid_out
               FROM bets WHERE status != 'pending'"""
        ).fetchone()
        open_row = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(stake), 0) AS at_risk FROM bets WHERE status = 'pending'"
        ).fetchone()
    handle, paid = float(row["handle"]), float(row["paid_out"])
    return {"settled_bets": int(row["bets"]), "handle": handle, "paid_out": paid, "house_profit": handle - paid,
            "hold_pct": (handle - paid) / handle if handle else 0.0,
            "open_bets": int(open_row["n"]), "open_stakes": float(open_row["at_risk"])}


# ---------------------------------------------------------------- bets
def insert_bet(conn: sqlite3.Connection, bet: dict) -> int:
    cols = list(bet.keys())
    cur = conn.execute(
        f"INSERT INTO bets ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [bet[c] for c in cols],
    )
    return int(cur.lastrowid)


def load_bets(username: str | None = None, status: str | None = None, db_path=None) -> pd.DataFrame:
    sql, params = "SELECT * FROM bets WHERE 1=1", []
    if username:
        sql += " AND username = ?"
        params.append(username)
    if status:
        sql += " AND status = ?"
        params.append(status)
    with connect(db_path) as conn:
        return pd.read_sql_query(sql + " ORDER BY bet_id DESC", conn, params=params)
