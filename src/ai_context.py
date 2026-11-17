"""Builds the data the AI analyst is grounded in (pure functions, no Streamlit).

Everything the chatbot may cite is collected here: how the models work, what the data
does and doesn't cover, ratings, upcoming games with lines and player props, futures,
league leaders, back-test results, the house ledgers and the user's own bets.
"""

from __future__ import annotations

import pandas as pd

from src import futures, players
from src.odds_model import STANDARD_JUICE, VIG, TeamRatings, make_lines

STAT_NAMES = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "3-pointers made"}


def model_description(r: TeamRatings, house_edge: bool) -> dict:
    return {
        "game_odds": ("Team power ratings from a weighted ridge regression of every completed game's point margin: "
                      "margin = home_court + rating_home - rating_away. Recent games weigh more (45-day half-life); "
                      "last season's games count 35%. P(home win) = NormalCDF(predicted margin / sigma), "
                      f"sigma = {r.sigma:.1f} points. Home-court advantage = {r.home_court:+.2f} points."),
        "totals": ("A second regression splits each team into offense (points scored vs. an average defense) and "
                   "defense (points allowed vs. an average offense). Predicted total = both teams' expected points. "
                   f"Game totals vary with sigma = {r.sigma_total:.1f} points."),
        "spreads": "Spread = predicted margin rounded to the nearest 0.5.",
        "player_props": ("Line = the player's recency-weighted average over their last 20 games (8-game half-life), "
                         "set at x.5 so there are no pushes. P(over) = 1 - NormalCDF((line - mean) / sd), using the "
                         "player's own game-to-game standard deviation. Offered for each team's top 4 players by "
                         "recent minutes who played the team's latest game."),
        "futures": ("5,000 simulated seasons: remaining games drawn from each team's chance to beat an average team, "
                    "then seeding, play-in and best-of-7 playoff series using the ratings model with home court. "
                    "Title probability = share of simulations won."),
        "pricing_now": (f"House edge ON: spreads, totals and props at {STANDARD_JUICE}, moneylines carry a "
                        f"{VIG:.1%} margin, futures about 25%. Bettors have negative expected value by design."
                        if house_edge else
                        "House edge OFF (fair odds): spreads, totals and props at +100, moneylines and futures at the "
                        "model's true probabilities. Expected value is about zero for both sides."),
        "parlays": "2-8 legs from different games; decimal odds multiply; pushed/voided legs drop out.",
    }


DATA_COVERAGE = {
    "included": ["Every NBA game of 2025-26 (incl. play-in and playoffs) and 2026-27 to date, from the ESPN API",
                 "Team box scores (points, rebounds, assists, shooting) for each game",
                 "About 40,000 player box scores (minutes, points, rebounds, assists, 3PM, shooting, +/-)",
                 "Schedule for the next 7 days; data refreshes every 30 minutes"],
    "not_included": ["Injuries, lineups, rest days and travel (the model can't adjust for a missing star)",
                     "Pace / possessions and advanced stats",
                     "Real sportsbook lines and news"],
}


def power_ratings(r: TeamRatings, standings: pd.DataFrame) -> list[dict]:
    st = standings.set_index("team") if not standings.empty else pd.DataFrame()
    out = []
    for rank, (team, rating) in enumerate(r.ratings.items(), start=1):
        rec = st.loc[team] if team in st.index else None
        out.append({"rank": rank, "team": team,
                    "conference": "East" if team in futures.EAST else "West",
                    "record": f"{int(rec['wins'])}-{int(rec['losses'])}" if rec is not None else None,
                    "net_rating": round(float(rating), 2),
                    "offense_pts": round(float(r.offense.get(team, r.league_avg_pts)), 1),
                    "defense_pts_allowed": round(float(r.defense.get(team, r.league_avg_pts)), 1)})
    return out


def upcoming_games(games: pd.DataFrame, r: TeamRatings, pg: pd.DataFrame, house_edge: bool, now: pd.Timestamp,
                   n_games: int = 10, n_with_props: int = 6) -> list[dict]:
    from src.betting import VOID_STATUSES
    up = games[~games["completed"] & ~games["status"].isin(VOID_STATUSES) & (games["tipoff_utc"] > now)]
    out = []
    for i, g in enumerate(up.sort_values("tipoff_utc").head(n_games).itertuples()):
        ln = make_lines(r, g.home_team, g.away_team, g.neutral_site, house_edge=house_edge)
        item = {
            "game": f"{g.away_team} @ {g.home_team}", "tipoff_utc": g.tipoff_utc.isoformat(),
            "model": {"home_win_prob": round(ln["p_home"], 3),
                      "projected_score": {g.away_team: round(ln["pred_away_pts"], 1),
                                          g.home_team: round(ln["pred_home_pts"], 1)}},
            "lines": {"moneyline": {g.away_team: ln["away_ml"], g.home_team: ln["home_ml"]},
                      "spread": {g.home_team: ln["home_spread"], g.away_team: ln["away_spread"],
                                 "price": ln["spread_odds"]},
                      "total": {"line": ln["total"], "price": ln["total_odds"]}},
        }
        if i < n_with_props and not pg.empty:
            menu = (players.prop_menu(pg, g.away_team, g.home_team, house_edge)
                    + players.prop_menu(pg, g.home_team, g.away_team, house_edge))
            item["player_props"] = [{"player": m["player_name"], "team": m["team"], "stat": STAT_NAMES[m["stat"]],
                                     "line": m["line"], "recent_avg": round(m["mean"], 1),
                                     "model_p_over": round(m["p_over"], 3), "price": m["over_odds"]} for m in menu]
        out.append(item)
    return out


def futures_board(sim: pd.DataFrame, house_edge: bool) -> list[dict]:
    board = futures.futures_odds(sim, house_edge)
    return [{"team": r.team, "conference": r.conference, "record": f"{r.wins}-{r.losses}",
             "projected_wins": round(r.proj_wins, 1), "playoff_prob": round(r.p_playoffs, 3),
             "finals_prob": round(r.p_conf, 3), "title_prob": round(r.p_title, 4), "odds": int(r.odds)}
            for r in board.itertuples()]


def player_leaders(pg: pd.DataFrame, season: int, n: int = 30) -> list[dict]:
    avg = players.season_averages(pg, season, min_games=5)
    if avg.empty:
        return []
    cols = ["name", "team", "position", "gp", "mpg", "ppg", "rpg", "apg", "tpm", "fg_pct", "fg3_pct"]
    keep = set(avg.nlargest(n, "ppg").index) | set(avg.nlargest(10, "rpg").index) | set(avg.nlargest(10, "apg").index)
    return avg.loc[sorted(keep, key=lambda i: -avg.loc[i, "ppg"]), cols].round(3).to_dict("records")


def user_betting(summary: dict, balance: float, bets: pd.DataFrame, parlays: pd.DataFrame,
                 legs: pd.DataFrame) -> dict:
    """The user's account. Stakes are split into open (still riding) and settled amounts."""
    def bet_rows(df: pd.DataFrame) -> list[dict]:
        cols = ["matchup", "bet_type", "selection", "line", "american_odds", "player_name", "prop_stat", "stake",
                "potential_payout", "status", "payout", "book"]
        return df[[c for c in cols if c in df]].to_dict("records")

    open_parlays = []
    for p in parlays[parlays["status"] == "pending"].to_dict("records"):
        pl = legs[legs["parlay_id"] == p["parlay_id"]]
        open_parlays.append({"stake": p["stake"], "decimal_odds": round(p["decimal_odds"], 2),
                             "potential_payout": p["potential_payout"], "book": p["book"],
                             "legs": pl[["matchup", "bet_type", "selection", "line", "american_odds", "player_name",
                                         "prop_stat", "result"]].to_dict("records")})
    return {
        "balance": round(balance, 2),
        "summary": summary,
        "note": ("open_stakes = money currently riding on pending wagers (already deducted from the balance); "
                 "settled_staked / settled_returned / profit / roi cover settled wagers only."),
        "open_bets": bet_rows(bets[bets["status"] == "pending"]),
        "open_parlays": open_parlays,
        "recent_settled_bets": bet_rows(bets[bets["status"] != "pending"].head(10)),
        "settled_parlays": parlays[parlays["status"] != "pending"].head(5)[
            ["n_legs", "decimal_odds", "stake", "status", "payout"]].to_dict("records"),
    }
