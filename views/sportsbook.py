"""Sportsbook: upcoming games, model-generated odds, and the bet slip."""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src import llm
from src.betting import VOID_STATUSES, BetError, place_bet
from src.data_fetcher import EASTERN
from src.odds_model import american_to_implied_prob, make_lines, payout_for
from src.ui import CURRENCY, fmt_line, fmt_odds, get_games, get_ratings, money, sidebar, stream_llm

games = get_games()
user = sidebar()
ratings = get_ratings(games)

st.title("🏀 Sportsbook")
st.caption("Odds are set by our ratings model (see 📊 Data Analysis → Model evaluation). "
           "Betting closes at tip-off; bets settle automatically when ESPN posts the final score.")

if msg := st.session_state.pop("bet_confirmation", None):
    st.toast(msg, icon="✅")
    st.success(msg + "  ·  See **🎟️ My Bets** to track it.", icon="✅")

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
team_filter = f2.multiselect("Filter by team", teams)

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
            "moneyline": {g.home_team: lines["home_ml"], g.away_team: lines["away_ml"]},
            "moneyline_implied_prob": {g.home_team: round(american_to_implied_prob(lines["home_ml"]), 3),
                                       g.away_team: round(american_to_implied_prob(lines["away_ml"]), 3)},
            "spread": {g.home_team: lines["home_spread"], g.away_team: lines["away_spread"], "price": -110},
            "total": {"line": lines["total"], "price": -110},
        },
    }


def game_card(g) -> None:
    """One game: odds, model projection, bet slip and AI preview."""
    lines = make_lines(ratings, g.home_team, g.away_team, g.neutral_site)
    started = g.tipoff_utc <= now
    with st.container(border=True):
        top = st.columns([3, 2, 2, 2])
        tip = g.tipoff_utc.tz_convert(EASTERN).strftime("%I:%M %p ET")
        top[0].markdown(f"**{g.away_team_name}** ({g.away_record or '-'})  \n"
                        f"@ **{g.home_team_name}** ({g.home_record or '-'})  \n"
                        f"{'🔴 LIVE, betting closed' if started else '🕒 ' + tip}")
        top[1].markdown(f"**Moneyline**  \n{g.away_team} {fmt_odds(lines['away_ml'])}  \n"
                        f"{g.home_team} {fmt_odds(lines['home_ml'])}")
        top[2].markdown(f"**Spread**  \n{g.away_team} {fmt_line(lines['away_spread'])} (-110)  \n"
                        f"{g.home_team} {fmt_line(lines['home_spread'])} (-110)")
        top[3].markdown(f"**Total**  \nO {lines['total']:g} (-110)  \nU {lines['total']:g} (-110)")
        fav, fav_p = ((g.home_team, lines["p_home"]) if lines["p_home"] >= 0.5
                      else (g.away_team, 1 - lines["p_home"]))
        st.caption(f"📈 Model: **{fav}** wins {fav_p:.0%} of the time · projected score "
                   f"{g.away_team} {lines['pred_away_pts']:.0f}, {g.home_team} {lines['pred_home_pts']:.0f}")

        if started:
            return
        bet_col, ai_col = st.columns(2)
        with bet_col.expander("🎟️ Place a bet"):
            if not user:
                st.info("Log in (sidebar) to bet.")
            else:
                market = st.radio("Market", ["moneyline", "spread", "total"], horizontal=True,
                                  key=f"mkt_{g.game_id}")
                if market == "moneyline":
                    opts = {f"{g.away_team} {fmt_odds(lines['away_ml'])}": (g.away_team, lines["away_ml"], None),
                            f"{g.home_team} {fmt_odds(lines['home_ml'])}": (g.home_team, lines["home_ml"], None)}
                elif market == "spread":
                    opts = {f"{g.away_team} {fmt_line(lines['away_spread'])} (-110)": (g.away_team, -110, lines["away_spread"]),
                            f"{g.home_team} {fmt_line(lines['home_spread'])} (-110)": (g.home_team, -110, lines["home_spread"])}
                else:
                    opts = {f"Over {lines['total']:g} (-110)": ("OVER", -110, lines["total"]),
                            f"Under {lines['total']:g} (-110)": ("UNDER", -110, lines["total"])}
                with st.form(f"bet_{g.game_id}"):
                    choice = st.radio("Selection", list(opts), key=f"sel_{g.game_id}_{market}")
                    stake = st.number_input(f"Stake ({CURRENCY})", min_value=1.0, value=50.0, step=10.0,
                                            key=f"stake_{g.game_id}")
                    sel, odds, line = opts[choice]
                    st.caption(f"Potential payout: {money(payout_for(stake, odds))} "
                               f"(profit {money(payout_for(stake, odds) - stake)})")
                    if st.form_submit_button("Place bet", type="primary"):
                        try:
                            bet_id = place_bet(user, g._asdict(), market, sel, odds, stake, line)
                            st.session_state["bet_confirmation"] = (
                                f"Bet #{bet_id} placed: **{choice}** ({g.away_team} @ {g.home_team}) "
                                f"for {money(stake)} {CURRENCY}")
                            st.rerun()  # refresh so the sidebar balance updates immediately
                        except BetError as exc:
                            st.error(str(exc))

        with ai_col.expander("🤖 AI matchup preview"):
            if st.button("Generate preview", key=f"ai_{g.game_id}", disabled=not llm.is_configured()):
                stream_llm(llm.game_preview, matchup_context(g, lines))
            elif not llm.is_configured():
                st.caption("Requires an Anthropic API key (see README).")


open_games = slate[slate["tipoff_utc"] > now]
live_games = slate[slate["tipoff_utc"] <= now]
if open_games.empty:
    st.info("Every game on this day has started. Pick another day above to bet.", icon="⏱️")
for g in open_games.itertuples():
    game_card(g)
if not live_games.empty:
    with st.expander(f"🔴 In progress or already started ({len(live_games)}): betting closed"):
        for g in live_games.itertuples():
            game_card(g)
