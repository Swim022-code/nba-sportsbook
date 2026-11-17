"""Bet placement and settlement with fake currency ("SimBucks").

Rules
-----
* Bets can only be placed before tip-off on games that aren't finished.
* Stake must be > 0 and <= the user's balance. The stake is deducted immediately.
* Settlement happens when the game is final:
    moneyline: selected team wins
    spread:    selected team's score + line > opponent's score  (equal = push)
    total:     OVER if combined score > line, UNDER if < line   (equal = push)
  Win  -> balance += stake * decimal_odds
  Push -> stake refunded
  Loss -> nothing returned
  Void -> game postponed/canceled: stake refunded
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src import database as db
from src.odds_model import payout_for

BET_TYPES = ("moneyline", "spread", "total")
VOID_STATUSES = {"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_SUSPENDED", "STATUS_FORFEIT"}
MAX_STAKE = 10_000


class BetError(ValueError):
    """A bet was rejected (bad stake, game started, etc.)."""


def place_bet(username: str, game: dict, bet_type: str, selection: str,
              american_odds: int, stake: float, line: float | None = None,
              now: datetime | None = None, db_path=None) -> int:
    """Validate and store a bet, deducting the stake. Returns the new bet_id."""
    now = now or datetime.now(timezone.utc)
    if bet_type not in BET_TYPES:
        raise BetError(f"Unknown bet type: {bet_type}")
    try:
        stake = round(float(stake), 2)
    except (TypeError, ValueError):
        raise BetError("Stake must be a number") from None
    if stake <= 0:
        raise BetError("Stake must be greater than 0")
    if stake > MAX_STAKE:
        raise BetError(f"Max stake is {MAX_STAKE:,} SimBucks")
    if game.get("completed"):
        raise BetError("That game is already final")
    tipoff = pd.Timestamp(game["tipoff_utc"])
    tipoff = tipoff.tz_localize("UTC") if tipoff.tzinfo is None else tipoff
    if tipoff <= pd.Timestamp(now):
        raise BetError("Betting is closed - that game has already tipped off")
    valid = {game["home_team"], game["away_team"]} if bet_type != "total" else {"OVER", "UNDER"}
    if selection not in valid:
        raise BetError(f"Selection must be one of {sorted(valid)}")
    if bet_type in ("spread", "total") and line is None:
        raise BetError("Spread and total bets need a line")

    with db.connect(db_path) as conn:
        row = conn.execute("SELECT balance FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            raise BetError("Unknown user - log in first")
        if stake > row["balance"]:
            raise BetError(f"Insufficient balance ({row['balance']:,.2f} SimBucks available)")
        conn.execute("UPDATE users SET balance = balance - ? WHERE username = ?", (stake, username))
        return db.insert_bet(conn, {
            "username": username,
            "game_id": str(game["game_id"]),
            "matchup": f"{game['away_team']} @ {game['home_team']}",
            "bet_type": bet_type,
            "selection": selection,
            "line": line,
            "american_odds": int(american_odds),
            "stake": stake,
            "potential_payout": payout_for(stake, int(american_odds)),
            "status": "pending",
            "placed_at": now.isoformat(timespec="seconds"),
        })


def grade_bet(bet: dict, home_team: str, home_score: float, away_score: float) -> str:
    """Return 'won', 'lost' or 'push' for a bet on a finished game."""
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


def settle_bets(games: pd.DataFrame, db_path=None) -> int:
    """Grade every pending bet whose game is final (or void it if the game was
    postponed/canceled). Returns the number of bets settled."""
    pending = db.load_bets(status="pending", db_path=db_path)
    if pending.empty or games.empty:
        return 0
    status = games["status"] if "status" in games else pd.Series("", index=games.index)
    final = games[games["completed"]].set_index("game_id")
    void_ids = set(games.loc[status.isin(VOID_STATUSES) & ~games["completed"], "game_id"])
    settled = 0
    with db.connect(db_path) as conn:
        for bet in pending.to_dict("records"):
            if bet["game_id"] in final.index:
                g = final.loc[bet["game_id"]]
                result = grade_bet(bet, g["home_team"], g["home_score"], g["away_score"])
            elif bet["game_id"] in void_ids:
                result = "void"
            else:
                continue
            payout = {"won": bet["potential_payout"], "push": bet["stake"],
                      "void": bet["stake"], "lost": 0.0}[result]
            conn.execute(
                "UPDATE bets SET status = ?, payout = ?, settled_at = ? WHERE bet_id = ?",
                (result, payout, db.now_iso(), bet["bet_id"]),
            )
            if payout:
                conn.execute("UPDATE users SET balance = balance + ? WHERE username = ?",
                             (payout, bet["username"]))
            settled += 1
    return settled


def betting_summary(bets: pd.DataFrame) -> dict:
    """Headline numbers for a user's bet history."""
    if bets.empty:
        return {"bets": 0, "pending": 0, "won": 0, "lost": 0, "push": 0, "void": 0,
                "staked": 0.0, "returned": 0.0, "profit": 0.0, "roi": 0.0, "win_rate": 0.0}
    settled = bets[bets["status"] != "pending"]
    staked = float(settled["stake"].sum())
    returned = float(settled["payout"].sum())
    decided = settled[settled["status"].isin(["won", "lost"])]
    return {
        "bets": len(bets),
        "pending": int((bets["status"] == "pending").sum()),
        "won": int((bets["status"] == "won").sum()),
        "lost": int((bets["status"] == "lost").sum()),
        "push": int((bets["status"] == "push").sum()),
        "void": int((bets["status"] == "void").sum()),
        "staked": staked,
        "returned": returned,
        "profit": returned - staked,
        "roi": (returned - staked) / staked if staked else 0.0,
        "win_rate": (decided["status"] == "won").mean() if len(decided) else 0.0,
    }
