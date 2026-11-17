"""Sportsbook: upcoming games, model-generated odds, and the bet slip."""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src import llm, style
from src.betting import VOID_STATUSES, BetError, place_bet
from src.data_fetcher import EASTERN
from src.odds_model import american_to_implied_prob, book_margin, make_lines, model_probability, payout_for
from src.ui import CURRENCY, fmt_line, fmt_odds, get_games, get_ratings, house_edge_on, money, sidebar, stream_llm

games = get_games()
user = sidebar()
ratings = get_ratings(games)
edge = house_edge_on()

style.hero("Sportsbook", "Odds set by our ratings model · betting closes at tip-off · bets settle automatically",
           "🏀", style.edge_badge(edge))

if msg := st.session_state.pop("bet_confirmation", None):
    st.toast(msg, icon="✅")
    st.success(msg + "  ·  Track it in **🎟️ My Bets**.", icon="✅")

now = pd.Timestamp(datetime.now(timezone.utc))
upcoming = games[~games["completed"] & ~games["status"].isin(VOID_STATUSES)].sort_values("tipoff_utc").copy()
if upcoming.empty:
    st.info("No upcoming games in the next 7 days (off-season?). Check back later.")
    st.stop()
upcoming["local_date"] = upcoming["tipoff_utc"].dt.tz_convert(EASTERN).dt.date

# ---- filters
f1, f2 = st.columns([1, 2])
days = sorted(upcoming["local_date"].unique())
open_days = sorted(upcoming.loc[upcoming["tipoff_utc"] > now, "local_date"].unique())
day = f1.selectbox("Game day (ET)", days, index=days.index(open_days[0]) if open_days else 0,
                   format_func=lambda d: d.strftime("%a %b %d"))
teams = sorted(set(upcoming["home_team"]) | set(upcoming["away_team"]))
team_filter = f2.multiselect("Filter by team", teams, placeholder="All teams")

slate = upcoming[upcoming["local_date"] == day]
if team_filter:
    slate = slate[slate["home_team"].isin(team_filter) | slate["away_team"].isin(team_filter)]
if slate.empty:
    st.info("No games match those filters.")
    st.stop()


def matchup_context(g, lines) -> dict:
    """Data handed to the LLM for a game preview."""
    def team(t):
        return {"net_rating": round(float(ratings.ratings.get(t, 0)), 2),
                "offense_pts": round(float(ratings.offense.get(t, 0)), 1),
                "defense_pts_allowed": round(float(ratings.defense.get(t, 0)), 1),
                "rating_rank": int(list(ratings.ratings.index).index(t) + 1) if t in ratings.ratings else None}
    return {
        "matchup": f"{g.away_team_name} ({g.away_record}) at {g.home_team_name} ({g.home_record})",
        "tipoff_et": g.tipoff_utc.tz_convert(EASTERN).strftime("%a %b %d %I:%M %p"),
        "home": {"team": g.home_team, **team(g.home_team)},
        "away": {"team": g.away_team, **team(g.away_team)},
        "model": {
            "home_court_advantage_pts": round(ratings.home_court, 2),
            "predicted_home_margin": round(float(lines["pred_margin"]), 1),
            "predicted_total": round(float(lines["pred_total"]), 1),
            "home_win_probability": round(lines["p_home"], 3),
            "margin_std_dev": round(ratings.sigma, 1),
        },
        "posted_lines": {
            "house_edge": "on (~4.5% margin)" if lines["house_edge"] else "off (fair odds)",
            "moneyline": {g.home_team: lines["home_ml"], g.away_team: lines["away_ml"]},
            "moneyline_implied_prob": {g.home_team: round(american_to_implied_prob(lines["home_ml"]), 3),
                                       g.away_team: round(american_to_implied_prob(lines["away_ml"]), 3)},
            "spread": {g.home_team: lines["home_spread"], g.away_team: lines["away_spread"],
                       "price": lines["spread_odds"]},
            "total": {"line": lines["total"], "price": lines["total_odds"]},
        },
    }


def bet_options(g, lines, market: str) -> dict:
    """label -> (selection, odds, line, model side)"""
    if market == "moneyline":
        return {f"{g.away_team} {fmt_odds(lines['away_ml'])}": (g.away_team, lines["away_ml"], None, "away"),
                f"{g.home_team} {fmt_odds(lines['home_ml'])}": (g.home_team, lines["home_ml"], None, "home")}
    if market == "spread":
        p = fmt_odds(lines["spread_odds"])
        return {f"{g.away_team} {fmt_line(lines['away_spread'])} ({p})": (g.away_team, lines["spread_odds"], lines["away_spread"], "away"),
                f"{g.home_team} {fmt_line(lines['home_spread'])} ({p})": (g.home_team, lines["spread_odds"], lines["home_spread"], "home")}
    p = fmt_odds(lines["total_odds"])
    return {f"Over {lines['total']:g} ({p})": ("OVER", lines["total_odds"], lines["total"], "over"),
            f"Under {lines['total']:g} ({p})": ("UNDER", lines["total_odds"], lines["total"], "under")}


def bet_slip(g, lines) -> None:
    gid = g.game_id
    market = st.radio("Market", ["moneyline", "spread", "total"], horizontal=True, key=f"mkt_{gid}",
                      format_func=str.title)
    opts = bet_options(g, lines, market)
    choice = st.radio("Pick", list(opts), key=f"sel_{gid}_{market}_{edge}")
    sel, odds, line, side = opts[choice]
    stake = st.number_input(f"Stake ({CURRENCY})", min_value=1.0, value=50.0, step=10.0, key=f"stake_{gid}")

    payout = payout_for(stake, odds)
    p_model = model_probability(ratings, lines, market, side)
    p_implied = american_to_implied_prob(odds)
    ev = p_model * (payout - stake) - (1 - p_model) * stake
    style.slip_summary(payout, payout - stake, p_model, p_implied, ev)
    st.caption("Model chance vs. what the odds imply: when the odds imply more than the model's chance, "
               "the bet has negative expected value. That gap is the house edge.")
    if st.button(f"Place bet: {choice}", type="primary", key=f"place_{gid}", width="stretch"):
        try:
            bet_id = place_bet(user, g._asdict(), market, sel, odds, stake, line)
            st.session_state["bet_confirmation"] = (
                f"Bet #{bet_id} placed: **{choice}** ({g.away_team} @ {g.home_team}) for {money(stake)} {CURRENCY}")
            st.rerun()  # refresh so the sidebar balance updates immediately
        except BetError as exc:
            st.error(str(exc))


def render_game(g) -> None:
    """One game: sportsbook card, then the bet slip and the AI preview."""
    lines = make_lines(ratings, g.home_team, g.away_team, g.neutral_site, house_edge=edge)
    started = g.tipoff_utc <= now
    tip = g.tipoff_utc.tz_convert(EASTERN).strftime("%a %I:%M %p ET")
    status = ('<span class="sb-live">● LIVE · betting closed</span>' if started
              else f'<span class="sb-time">🕒 {tip}</span>')
    margin = book_margin(lines["home_ml"], lines["away_ml"])
    foot = (f"Projected score: {g.away_team} {lines['pred_away_pts']:.0f}, {g.home_team} {lines['pred_home_pts']:.0f}"
            f" · Moneyline house margin {margin:.1%}")
    style.game_card(g, lines, status, foot)
    if started:
        return
    bet_col, ai_col = st.columns(2)
    with bet_col.expander("🎟️ Place a bet"):
        if user:
            bet_slip(g, lines)
        else:
            st.info("Log in (sidebar) to bet.")
    with ai_col.expander("🤖 AI matchup preview"):
        if st.button("Generate preview", key=f"ai_{g.game_id}", disabled=not llm.is_configured()):
            stream_llm(llm.game_preview, matchup_context(g, lines))
        elif not llm.is_configured():
            st.caption("Requires an Anthropic API key (see README).")
    st.write("")


open_games = slate[slate["tipoff_utc"] > now]
live_games = slate[slate["tipoff_utc"] <= now]
if open_games.empty:
    st.info("Every game on this day has started. Pick another day above to bet.", icon="⏱️")
for g in open_games.itertuples():
    render_game(g)
if not live_games.empty:
    with st.expander(f"🔴 In progress or already started ({len(live_games)}): betting closed"):
        for g in live_games.itertuples():
            render_game(g)
