"""Bet placement and settlement with fake currency ("SimBucks").

Wager types
-----------
* Singles: moneyline, spread, total, player prop (over/under), futures (champion).
* Parlays: 2-8 legs from different games (game lines and/or props). Odds multiply;
  every leg must win. A pushed or voided leg drops out (its odds count as 1.0).

Rules
-----
* Game bets close at tip-off. Stake must be > 0, <= MAX_STAKE and <= the balance.
  The stake is deducted immediately (it goes to the house).
* Settlement:
    moneyline: selected team wins
    spread:    selected team's score + line > opponent's score  (equal = push)
    total:     OVER if combined score > line, UNDER if < line   (equal = push)
    prop:      player's stat vs. line; player didn't play -> void
    future:    selected team wins the NBA Finals
  Win  -> balance += stake * decimal_odds      Push / Void -> stake refunded      Loss -> nothing
* Every wager records the `book` it was priced in ('edge' = house margin on, 'fair' = off),
  which keeps two separate house ledgers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import prod

import pandas as pd

from src import database as db
from src.odds_model import american_to_decimal, payout_for

GAME_BET_TYPES = ("moneyline", "spread", "total", "prop")
VOID_STATUSES = {"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_SUSPENDED", "STATUS_FORFEIT"}
PROP_STATS = ("pts", "reb", "ast", "fg3m")
MAX_STAKE = 10_000
MIN_LEGS, MAX_LEGS = 2, 8


class BetError(ValueError):
    """A bet was rejected (bad stake, game started, etc.)."""


# ---------------------------------------------------------------- validation
def _clean_stake(stake) -> float:
    try:
        stake = round(float(stake), 2)
    except (TypeError, ValueError):
        raise BetError("Stake must be a number") from None
    if stake <= 0:
        raise BetError("Stake must be greater than 0")
    if stake > MAX_STAKE:
        raise BetError(f"Max stake is {MAX_STAKE:,} SimBucks")
    return stake


def _check_book(book: str) -> None:
    if book not in db.BOOKS:
        raise BetError(f"Unknown book: {book}")


def validate_leg(game: dict, leg: dict, now: datetime) -> None:
    """Raise BetError unless `leg` is a valid, still-open bet on `game`."""
    kind, sel = leg["bet_type"], leg["selection"]
    if kind not in GAME_BET_TYPES:
        raise BetError(f"Unknown bet type: {kind}")
    if game.get("completed"):
        raise BetError("That game is already final")
    tipoff = pd.Timestamp(game["tipoff_utc"])
    tipoff = tipoff.tz_localize("UTC") if tipoff.tzinfo is None else tipoff
    if tipoff <= pd.Timestamp(now):
        raise BetError("Betting is closed: that game has already tipped off")
    valid = {"OVER", "UNDER"} if kind in ("total", "prop") else {game["home_team"], game["away_team"]}
    if sel not in valid:
        raise BetError(f"Selection must be one of {sorted(valid)}")
    if kind in ("spread", "total", "prop") and leg.get("line") is None:
        raise BetError("Spread, total and prop bets need a line")
    if kind == "prop" and (not leg.get("player_id") or leg.get("prop_stat") not in PROP_STATS):
        raise BetError("Prop bets need a player and a stat")


def _debit(conn, username: str, stake: float) -> None:
    row = conn.execute("SELECT balance FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        raise BetError("Unknown user: log in first")
    if stake > row["balance"]:
        raise BetError(f"Insufficient balance ({row['balance']:,.2f} SimBucks available)")
    conn.execute("UPDATE users SET balance = balance - ? WHERE username = ?", (stake, username))


# ---------------------------------------------------------------- placing wagers
def place_bet(username: str, game: dict, bet_type: str, selection: str, american_odds: int, stake: float,
              line: float | None = None, book: str = "edge", player_id: str | None = None,
              player_name: str | None = None, prop_stat: str | None = None,
              now: datetime | None = None, db_path=None) -> int:
    """Validate and store a single game bet (incl. props). Returns the new bet_id."""
    now = now or datetime.now(timezone.utc)
    _check_book(book)
    stake = _clean_stake(stake)
    leg = {"bet_type": bet_type, "selection": selection, "line": line, "player_id": player_id,
           "prop_stat": prop_stat}
    validate_leg(game, leg, now)
    with db.connect(db_path) as conn:
        _debit(conn, username, stake)
        return db.insert_bet(conn, {
            "username": username, "game_id": str(game["game_id"]),
            "matchup": f"{game['away_team']} @ {game['home_team']}",
            "bet_type": bet_type, "selection": selection, "line": line, "american_odds": int(american_odds),
            "stake": stake, "potential_payout": payout_for(stake, int(american_odds)), "status": "pending",
            "placed_at": now.isoformat(timespec="seconds"), "book": book,
            "player_id": player_id, "player_name": player_name, "prop_stat": prop_stat,
        })


def place_future(username: str, team: str, season: int, american_odds: int, stake: float, book: str = "edge",
                 now: datetime | None = None, db_path=None) -> int:
    """Bet on `team` to win the NBA championship for `season`."""
    now = now or datetime.now(timezone.utc)
    _check_book(book)
    stake = _clean_stake(stake)
    with db.connect(db_path) as conn:
        _debit(conn, username, stake)
        return db.insert_bet(conn, {
            "username": username, "game_id": f"FUT-{season}",
            "matchup": f"NBA Champion {season - 1}-{str(season)[-2:]}", "bet_type": "future",
            "selection": team, "line": None, "american_odds": int(american_odds), "stake": stake,
            "potential_payout": payout_for(stake, int(american_odds)), "status": "pending",
            "placed_at": now.isoformat(timespec="seconds"), "book": book,
        })


def parlay_odds(legs: list[dict]) -> float:
    """Decimal odds of a parlay = product of each leg's decimal odds."""
    return prod(american_to_decimal(int(leg["american_odds"])) for leg in legs)


def place_parlay(username: str, legs: list[dict], games: dict[str, dict], stake: float, book: str = "edge",
                 now: datetime | None = None, db_path=None) -> int:
    """Store a parlay. `legs` need game_id, bet_type, selection, line, american_odds (+ player fields
    for props); `games` maps game_id -> game dict for validation."""
    now = now or datetime.now(timezone.utc)
    _check_book(book)
    stake = _clean_stake(stake)
    if not MIN_LEGS <= len(legs) <= MAX_LEGS:
        raise BetError(f"A parlay needs {MIN_LEGS}-{MAX_LEGS} legs")
    ids = [str(leg["game_id"]) for leg in legs]
    if len(set(ids)) != len(ids):
        raise BetError("Only one leg per game (same-game parlays aren't offered)")
    for leg in legs:
        if leg["game_id"] not in games:
            raise BetError("A leg's game is no longer on the board")
        validate_leg(games[leg["game_id"]], leg, now)
    dec = parlay_odds(legs)
    with db.connect(db_path) as conn:
        _debit(conn, username, stake)
        cur = conn.execute(
            "INSERT INTO parlays (username, n_legs, decimal_odds, stake, potential_payout, status, placed_at, book) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
            (username, len(legs), dec, stake, round(stake * dec, 2), now.isoformat(timespec="seconds"), book))
        pid = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO parlay_legs (parlay_id, game_id, matchup, bet_type, selection, line, american_odds, "
            "player_id, player_name, prop_stat) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(pid, leg["game_id"], leg.get("matchup"), leg["bet_type"], leg["selection"], leg.get("line"),
              int(leg["american_odds"]), leg.get("player_id"), leg.get("player_name"), leg.get("prop_stat"))
             for leg in legs])
    return pid


# ---------------------------------------------------------------- grading
def grade_bet(bet: dict, home_team: str, home_score: float, away_score: float) -> str:
    """Return 'won', 'lost' or 'push' for a game-line bet on a finished game."""
    sel, kind, line = bet["selection"], bet["bet_type"], bet.get("line")
    if kind == "total":
        total = home_score + away_score
        if total == line:
            return "push"
        return "won" if (total > line) == (sel == "OVER") else "lost"
    picked_home = sel == home_team
    mine, theirs = (home_score, away_score) if picked_home else (away_score, home_score)
    if kind == "moneyline":
        return "won" if mine > theirs else "lost"
    adjusted = mine + line  # spread
    if adjusted == theirs:
        return "push"
    return "won" if adjusted > theirs else "lost"


class _Results:
    """Everything needed to decide whether a wager has been decided."""

    def __init__(self, games: pd.DataFrame, db_path=None):
        status = games["status"] if "status" in games else pd.Series("", index=games.index)
        self.finals = games[games["completed"]].set_index("game_id")
        self.voids = set(games.loc[status.isin(VOID_STATUSES) & ~games["completed"], "game_id"])
        self.games, self.db_path = games, db_path
        self._fetched, self._champions = None, {}

    def _box_fetched(self, game_id: str) -> bool:
        if self._fetched is None:
            self._fetched = db.fetched_box_ids(self.db_path)
        return game_id in self._fetched

    def _player_stat(self, game_id: str, player_id: str, stat: str):
        with db.connect(self.db_path) as conn:
            row = conn.execute(f"SELECT dnp, {stat} AS v FROM player_games WHERE game_id = ? AND player_id = ?",
                               (game_id, str(player_id))).fetchone()
        return None if row is None or row["dnp"] or row["v"] is None else float(row["v"])

    def _champion(self, season: int):
        if season not in self._champions and "season_type" in self.games:
            from src.futures import champion  # local import: futures depends on the odds model
            self._champions[season] = champion(self.games, season)
        return self._champions.get(season)

    def resolve(self, w: dict) -> str | None:
        """'won' | 'lost' | 'push' | 'void', or None if not decided yet."""
        kind, gid = w["bet_type"], str(w["game_id"])
        if kind == "future":
            champ = self._champion(int(gid.split("-")[1]))
            return None if champ is None else ("won" if w["selection"] == champ else "lost")
        if gid in self.voids:
            return "void"
        if gid not in self.finals.index:
            return None
        if kind == "prop":
            if not self._box_fetched(gid):
                return None  # box score not downloaded yet
            value = self._player_stat(gid, w["player_id"], w["prop_stat"])
            if value is None:
                return "void"  # player didn't play
            if value == w["line"]:
                return "push"
            return "won" if (value > w["line"]) == (w["selection"] == "OVER") else "lost"
        g = self.finals.loc[gid]
        return grade_bet(w, g["home_team"], g["home_score"], g["away_score"])


def settle_bets(games: pd.DataFrame, db_path=None) -> int:
    """Settle every decided single bet and parlay. Returns how many were settled."""
    pending = db.load_bets(status="pending", db_path=db_path)
    parlays = db.load_parlays(status="pending", db_path=db_path)
    if (pending.empty and parlays.empty) or games.empty:
        return 0
    res = _Results(games, db_path)
    settled = 0
    with db.connect(db_path) as conn:
        def pay(table: str, key: str, wid: int, status: str, payout: float, username: str):
            conn.execute(f"UPDATE {table} SET status = ?, payout = ?, settled_at = ? WHERE {key} = ?",
                         (status, payout, db.now_iso(), wid))
            if payout:
                conn.execute("UPDATE users SET balance = balance + ? WHERE username = ?", (payout, username))

        for bet in pending.to_dict("records"):
            result = res.resolve(bet)
            if result is None:
                continue
            payout = {"won": bet["potential_payout"], "push": bet["stake"], "void": bet["stake"], "lost": 0.0}[result]
            pay("bets", "bet_id", bet["bet_id"], result, payout, bet["username"])
            settled += 1

        if not parlays.empty:
            legs = db.load_parlay_legs(parlays["parlay_id"].tolist(), db_path)
            for p in parlays.to_dict("records"):
                plegs = legs[legs["parlay_id"] == p["parlay_id"]].to_dict("records")
                results = []
                for leg in plegs:
                    r = leg["result"] if leg["result"] != "pending" else res.resolve(leg)
                    if r and leg["result"] == "pending":
                        conn.execute("UPDATE parlay_legs SET result = ? WHERE leg_id = ?", (r, leg["leg_id"]))
                    results.append(r)
                if "lost" in results:
                    pay("parlays", "parlay_id", p["parlay_id"], "lost", 0.0, p["username"])
                elif all(r is not None for r in results):
                    winners = [leg for leg, r in zip(plegs, results, strict=True) if r == "won"]
                    if winners:
                        payout = round(p["stake"] * parlay_odds(winners), 2)
                        pay("parlays", "parlay_id", p["parlay_id"], "won", payout, p["username"])
                    else:  # every leg pushed or was voided
                        pay("parlays", "parlay_id", p["parlay_id"], "push", p["stake"], p["username"])
                else:
                    continue
                settled += 1
    return settled


def betting_summary(wagers: pd.DataFrame) -> dict:
    """Headline numbers for a user's wagers (singles and/or parlays).

    open_stakes    = money riding on pending wagers (already deducted from the balance)
    settled_staked = money staked on wagers that have settled; profit and ROI use settled wagers only
    """
    if wagers.empty:
        return {"bets": 0, "pending": 0, "won": 0, "lost": 0, "push": 0, "void": 0, "open_stakes": 0.0,
                "settled_staked": 0.0, "settled_returned": 0.0, "profit": 0.0, "roi": 0.0, "win_rate": 0.0}
    pending = wagers[wagers["status"] == "pending"]
    settled = wagers[wagers["status"] != "pending"]
    staked = float(settled["stake"].sum())
    returned = float(settled["payout"].sum())
    decided = settled[settled["status"].isin(["won", "lost"])]
    return {
        "bets": len(wagers),
        "pending": len(pending),
        "won": int((wagers["status"] == "won").sum()),
        "lost": int((wagers["status"] == "lost").sum()),
        "push": int((wagers["status"] == "push").sum()),
        "void": int((wagers["status"] == "void").sum()),
        "open_stakes": float(pending["stake"].sum()),
        "settled_staked": staked,
        "settled_returned": returned,
        "profit": returned - staked,
        "roi": (returned - staked) / staked if staked else 0.0,
        "win_rate": float((decided["status"] == "won").mean()) if len(decided) else 0.0,
    }
