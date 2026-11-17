"""Futures: bet on the NBA champion, priced by a 5,000-run season simulation."""

import plotly.graph_objects as go
import streamlit as st

from src import futures, style
from src.betting import BetError, place_future
from src.charts import BLUE, styled
from src.odds_model import american_to_implied_prob, payout_for
from src.ui import (
    CURRENCY,
    current_book,
    fmt_odds,
    get_games,
    get_ratings,
    get_title_sim,
    house_edge_on,
    money,
    season_label,
    sidebar,
)

games = get_games()
user = sidebar()
ratings = get_ratings(games)
edge = house_edge_on()
season = int(games["season"].max())

style.hero("Futures", f"Who wins the {season_label(season)} NBA title? Odds from 5,000 simulated seasons", "🏆",
           style.edge_badge(edge))

if msg := st.session_state.pop("future_confirmation", None):
    st.success(msg + "  ·  Track it in **🎟️ My Bets**. It settles when the Finals end.", icon="✅")

sim = get_title_sim(games, ratings, season)
board = futures.futures_odds(sim, edge)
last_champ = futures.champion(games, season - 1)

c = st.columns(4)
c[0].metric("Title favorite", board.iloc[0]["team"], f"{board.iloc[0]['p_title']:.0%} to win it all", delta_color="off")
c[1].metric("Teams with 5%+ title odds", int((board["p_title"] >= 0.05).sum()))
c[2].metric("Market margin", f"{(sum(american_to_implied_prob(o) for o in board['odds']) - 1):.0%}",
            help="Sum of implied probabilities minus 100%. Real books hold far more on futures than on single games.")
c[3].metric(f"{season_label(season - 1)} champion", last_champ or "-")

left, right = st.columns([3, 2], gap="large")
with left:
    top = board.head(12).iloc[::-1]
    fig = go.Figure()
    # Winning the title is a subset of reaching the Finals: draw Finals (light) behind title (dark).
    fig.add_bar(y=top["team"], x=top["p_conf"], orientation="h", name="Reach the Finals", marker_color="#b7d3f6",
                hovertemplate="%{y}: %{x:.1%} to reach the Finals<extra></extra>")
    fig.add_bar(y=top["team"], x=top["p_title"], orientation="h", name="Win the title", marker_color=BLUE,
                width=0.5, hovertemplate="%{y}: %{x:.1%} to win the title<extra></extra>")
    fig.update_layout(barmode="overlay", height=430, xaxis_tickformat=".0%", xaxis_title="Probability")
    st.plotly_chart(styled(fig, "Title and Finals chances (top 12)"), width="stretch")

with right:
    st.subheader("🎟️ Bet on the champion")
    labels = {r.team: f"{r.team}  {fmt_odds(r.odds)}  ({r.p_title:.1%})" for r in board.itertuples()}
    team = st.selectbox("Team (odds, model chance)", list(labels), format_func=labels.get)
    row = board.set_index("team").loc[team]
    stake = st.number_input(f"Stake ({CURRENCY})", min_value=1.0, value=20.0, step=5.0)
    odds = int(row["odds"])
    payout = payout_for(stake, odds)
    p = float(row["p_title"])
    style.slip_summary(payout, payout - stake, p, american_to_implied_prob(odds),
                       p * (payout - stake) - (1 - p) * stake)
    if st.button(f"Place bet: {team} to win the title ({fmt_odds(odds)})", type="primary", width="stretch",
                 disabled=not user):
        try:
            bid = place_future(user, team, season, odds, stake, book=current_book())
            st.session_state["future_confirmation"] = f"Futures bet #{bid}: **{team}** {fmt_odds(odds)} for {money(stake)} {CURRENCY}"
            st.rerun()
        except BetError as exc:
            st.error(str(exc))
    if not user:
        st.caption("Log in (sidebar) to bet.")

st.subheader("Projected standings & odds")
table = board.assign(logo=board["team"].map(style.logo_url), record=board["wins"].astype(str) + "-" + board["losses"].astype(str))
st.dataframe(
    table[["logo", "team", "conference", "record", "rating", "proj_wins", "p_playoffs", "p_conf", "p_title", "odds"]],
    hide_index=True, width="stretch", row_height=38,
    column_config={
        "logo": st.column_config.ImageColumn(" ", width="small"), "team": "Team", "conference": "Conf",
        "record": "W-L", "rating": st.column_config.NumberColumn("Net rating", format="%+.1f"),
        "proj_wins": st.column_config.NumberColumn("Proj. wins", format="%.1f"),
        "p_playoffs": st.column_config.ProgressColumn("Playoffs", format="percent", min_value=0, max_value=1),
        "p_conf": st.column_config.NumberColumn("Reach Finals", format="percent"),
        "p_title": st.column_config.NumberColumn("Win title", format="percent"),
        "odds": st.column_config.NumberColumn("Title odds", format="%+d"),
    },
)
with st.expander("How the simulation works"):
    st.markdown(f"""
1. **Rest of the regular season.** Each team adds wins drawn from a binomial distribution: its games left
   × its chance to beat an average team (from the power ratings, half home and half away).
2. **Seeding.** Teams are ranked by wins in each conference. Seeds 7-10 play the **play-in**.
3. **Playoffs.** Best-of-7 series, with home court for the higher seed. Each game uses the ratings model
   (home-court edge {ratings.home_court:+.1f} pts, σ = {ratings.sigma:.1f}).
4. This runs **5,000 times**. A team's title chance is the share of simulations it wins.

With the house edge on, futures odds are shaded so the implied probabilities add up to about 125%. That's
typical for real books, which hold much more on futures than on single games.
""")
