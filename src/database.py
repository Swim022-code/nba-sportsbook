"""SQLite storage: cached games and player box scores, user wallets, bets and parlays.

Everything lives in one local file (data/sportsbook.db), so the app needs no
database server. `init_db()` creates missing tables and upgrades older databases
in place (adds new columns), so existing users and bets are kept.

Money model
-----------
A stake leaves the bettor's balance when the bet is placed and goes to "the house".
When it settles, the house pays back stake + winnings (won), the stake (push/void),
or nothing (lost). Each wager is tagged with the `book` it was priced in:
'edge' (house margin on) or 'fair' (no margin), and each book has its own ledger.
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
BOOKS = ("edge", "fair")

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
    game_id TEXT NOT NULL,       -- ESPN game id, or FUT-<season> for futures
    matchup TEXT,
    bet_type TEXT NOT NULL,      -- moneyline | spread | total | prop | future
    selection TEXT NOT NULL,     -- team abbreviation, or OVER / UNDER
    line REAL,                   -- spread for the selected team, total, or prop line
    american_odds INTEGER NOT NULL,
    stake REAL NOT NULL,
    potential_payout REAL NOT NULL,  -- stake + profit if it wins
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | won | lost | push | void
    payout REAL DEFAULT 0,
    placed_at TEXT,
    settled_at TEXT,
    book TEXT DEFAULT 'edge',    -- edge | fair
    player_id TEXT,              -- props only
    player_name TEXT,
    prop_stat TEXT               -- pts | reb | ast | fg3m
);
CREATE TABLE IF NOT EXISTS parlays (
    parlay_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL REFERENCES users(username),
    n_legs INTEGER NOT NULL,
    decimal_odds REAL NOT NULL,
    stake REAL NOT NULL,
    potential_payout REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    payout REAL DEFAULT 0,
    placed_at TEXT,
    settled_at TEXT,
    book TEXT DEFAULT 'edge'
);
CREATE TABLE IF NOT EXISTS parlay_legs (
    leg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    parlay_id INTEGER NOT NULL REFERENCES parlays(parlay_id),
    game_id TEXT NOT NULL,
    matchup TEXT,
    bet_type TEXT NOT NULL,
    selection TEXT NOT NULL,
    line REAL,
    american_odds INTEGER NOT NULL,
    player_id TEXT,
    player_name TEXT,
    prop_stat TEXT,
    result TEXT DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS player_games (
    game_id TEXT, player_id TEXT, season INTEGER, game_date TEXT,
    team TEXT, opponent TEXT, is_home INTEGER,
    name TEXT, position TEXT, headshot TEXT, starter INTEGER, dnp INTEGER, dnp_reason TEXT,
    minutes REAL, pts REAL, reb REAL, ast REAL, fg3m REAL, fg3a REAL, fgm REAL, fga REAL,
    ftm REAL, fta REAL, stl REAL, blk REAL, tov REAL, plus_minus REAL,
    PRIMARY KEY (game_id, player_id)
);
CREATE TABLE IF NOT EXISTS box_fetched (
    game_id TEXT PRIMARY KEY,
    fetched_at TEXT
);
"""

# Columns added after the first release; init_db() adds them to older databases.
MIGRATIONS = {
    "bets": {"book": "TEXT DEFAULT 'edge'", "player_id": "TEXT", "player_name": "TEXT", "prop_stat": "TEXT"},
}

GAME_COLUMNS = [
    "game_id", "tipoff_utc", "game_date", "season", "season_type", "status", "completed",
    "neutral_site", "venue", "attendance",
    *[f"{side}_{c}" for side in ("home", "away") for c in (
        "team", "team_name", "score", "record", "pts", "reb", "ast", "fgm", "fga", "fg_pct",
        "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct")],
]
PLAYER_COLUMNS = [
    "game_id", "player_id", "season", "game_date", "team", "opponent", "is_home", "name", "position", "headshot",
    "starter", "dnp", "dnp_reason", "minutes", "pts", "reb", "ast", "fg3m", "fg3a", "fgm", "fga", "ftm", "fta",
    "stl", "blk", "tov", "plus_minus",
]

# All money-moving wagers (single bets and parlays) as one view.
WAGERS_SQL = """
    SELECT username, book, stake, payout, status, placed_at, settled_at, 'single' AS kind FROM bets
    UNION ALL
    SELECT username, book, stake, payout, status, placed_at, settled_at, 'parlay' AS kind FROM parlays
"""


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
    """Create tables, and upgrade databases created by older versions of the app."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            for col, decl in cols.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
                    if (table, col) == ("bets", "book"):
                        # Before ledgers existed, fair-mode spread/total bets were priced at +100.
                        conn.execute("UPDATE bets SET book = 'fair' WHERE bet_type IN ('spread', 'total') "
                                     "AND american_odds = 100")


# ---------------------------------------------------------------- games cache
def _upsert(table: str, df: pd.DataFrame, columns: list[str], db_path=None, stamp: bool = False) -> int:
    if df.empty:
        return 0
    df = df.reindex(columns=columns).copy()
    if stamp:
        df["fetched_at"] = now_iso()
    df = df.astype(object).where(df.notna(), None)
    cols = list(df.columns)
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    with connect(db_path) as conn:
        conn.executemany(sql, df.itertuples(index=False, name=None))
    return len(df)


def upsert_games(df: pd.DataFrame, db_path=None) -> int:
    return _upsert("games", df, GAME_COLUMNS, db_path, stamp=True)


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


# ---------------------------------------------------------------- player box scores
def upsert_player_games(df: pd.DataFrame, game_ids: list[str], db_path=None) -> int:
    """Store player rows and mark their games as fetched (even games with no rows)."""
    n = _upsert("player_games", df, PLAYER_COLUMNS, db_path)
    with connect(db_path) as conn:
        conn.executemany("INSERT OR REPLACE INTO box_fetched VALUES (?, ?)", [(g, now_iso()) for g in game_ids])
    return n


def fetched_box_ids(db_path=None) -> set[str]:
    with connect(db_path) as conn:
        return {r["game_id"] for r in conn.execute("SELECT game_id FROM box_fetched")}


def load_player_games(db_path=None) -> pd.DataFrame:
    with connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM player_games", conn)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["dnp"] = df["dnp"].astype(bool)
        df["starter"] = df["starter"].astype(bool)
    return df


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
        conn.execute("DELETE FROM parlay_legs WHERE parlay_id IN (SELECT parlay_id FROM parlays WHERE username = ?)",
                     (username,))
        conn.execute("DELETE FROM parlays WHERE username = ?", (username,))
        conn.execute("UPDATE users SET balance = ? WHERE username = ?", (STARTING_BALANCE, username))


def leaderboard(db_path=None) -> pd.DataFrame:
    """Every user with their balance, net result and betting record (singles + parlays)."""
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            f"""SELECT u.username, u.balance, u.created_at,
                       COUNT(w.status) AS wagers,
                       SUM(CASE WHEN w.status = 'won' THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN w.status = 'lost' THEN 1 ELSE 0 END) AS losses,
                       SUM(CASE WHEN w.kind = 'parlay' THEN 1 ELSE 0 END) AS parlays,
                       COALESCE(SUM(CASE WHEN w.status = 'pending' THEN w.stake END), 0) AS open_stakes,
                       COALESCE(SUM(CASE WHEN w.status != 'pending' THEN w.stake END), 0) AS settled_staked,
                       COALESCE(SUM(CASE WHEN w.status != 'pending' THEN w.payout END), 0) AS settled_returned,
                       COALESCE(MAX(CASE WHEN w.status = 'won' THEN w.payout - w.stake END), 0) AS biggest_win
                FROM users u LEFT JOIN ({WAGERS_SQL}) w ON w.username = u.username
                GROUP BY u.username""",
            conn,
        )
    if df.empty:
        return df
    num = ["balance", "wagers", "wins", "losses", "parlays", "open_stakes", "settled_staked", "settled_returned",
           "biggest_win"]
    df[num] = df[num].apply(pd.to_numeric).fillna(0).astype(float)
    df["net"] = df["balance"] + df["open_stakes"] - STARTING_BALANCE
    staked = df["settled_staked"].where(df["settled_staked"] > 0)
    df["roi"] = ((df["settled_returned"] - df["settled_staked"]) / staked).astype(float)
    decided = (df["wins"] + df["losses"]).where(lambda x: x > 0)
    df["win_rate"] = (df["wins"] / decided).astype(float)
    df[["wagers", "wins", "losses", "parlays"]] = df[["wagers", "wins", "losses", "parlays"]].astype(int)
    df = df.sort_values(["net", "balance"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------- the house
def house_summary(book: str = "edge", db_path=None) -> dict:
    """The house's side of every wager priced in `book` (edge or fair).

    house profit = stakes taken on settled wagers - money paid back to bettors.
    """
    with connect(db_path) as conn:
        row = conn.execute(
            f"""SELECT COUNT(*) AS n, COALESCE(SUM(stake), 0) AS handle, COALESCE(SUM(payout), 0) AS paid_out
                FROM ({WAGERS_SQL}) WHERE status != 'pending' AND book = ?""", (book,)).fetchone()
        open_row = conn.execute(
            f"""SELECT COUNT(*) AS n, COALESCE(SUM(stake), 0) AS at_risk
                FROM ({WAGERS_SQL}) WHERE status = 'pending' AND book = ?""", (book,)).fetchone()
    handle, paid = float(row["handle"]), float(row["paid_out"])
    return {"book": book, "settled_bets": int(row["n"]), "handle": handle, "paid_out": paid,
            "house_profit": handle - paid, "hold_pct": (handle - paid) / handle if handle else 0.0,
            "open_bets": int(open_row["n"]), "open_stakes": float(open_row["at_risk"])}


def house_timeline(db_path=None) -> pd.DataFrame:
    """Cumulative house profit per book, in settlement order (for the edge-vs-fair chart)."""
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            f"""SELECT book, settled_at, stake - payout AS house_pnl FROM ({WAGERS_SQL})
                WHERE status != 'pending' ORDER BY settled_at""", conn)
    if df.empty:
        return df
    df["settled_at"] = pd.to_datetime(df["settled_at"], utc=True, format="ISO8601")
    df["n"] = df.groupby("book").cumcount() + 1
    df["house_profit"] = df.groupby("book")["house_pnl"].cumsum()
    return df


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


def load_parlays(username: str | None = None, status: str | None = None, db_path=None) -> pd.DataFrame:
    sql, params = "SELECT * FROM parlays WHERE 1=1", []
    if username:
        sql += " AND username = ?"
        params.append(username)
    if status:
        sql += " AND status = ?"
        params.append(status)
    with connect(db_path) as conn:
        return pd.read_sql_query(sql + " ORDER BY parlay_id DESC", conn, params=params)


def load_parlay_legs(parlay_ids: list[int] | None = None, db_path=None) -> pd.DataFrame:
    with connect(db_path) as conn:
        if parlay_ids is None:
            return pd.read_sql_query("SELECT * FROM parlay_legs", conn)
        if not parlay_ids:
            return pd.read_sql_query("SELECT * FROM parlay_legs WHERE 0", conn)
        marks = ", ".join("?" * len(parlay_ids))
        return pd.read_sql_query(f"SELECT * FROM parlay_legs WHERE parlay_id IN ({marks})", conn,
                                 params=[int(i) for i in parlay_ids])
