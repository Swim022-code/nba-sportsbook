"""Leaderboard: every user ranked, plus the house's two ledgers (edge vs. fair odds)."""

import plotly.graph_objects as go
import streamlit as st

from src import database as db
from src import style
from src.charts import BLUE, GRAY, RED, styled
from src.ui import CURRENCY, current_book, get_games, money, sidebar

get_games()  # refresh + settle so standings are current
user = sidebar()

style.hero("Leaderboard", "Every player ranked by profit, and how the house is doing in each odds mode", "🏆")

lb = db.leaderboard()
if lb.empty:
    st.info("No accounts yet. Log in from the sidebar to join.")
    st.stop()

c = st.columns(4)
c[0].metric("Players", len(lb))
c[1].metric("Wagers placed", int(lb["wagers"].sum()))
c[2].metric("In winning position", int((lb["net"] > 0).sum()))
c[3].metric("On open bets", money(lb["open_stakes"].sum()))

style.podium(lb.head(3).to_dict("records"), CURRENCY)
st.write("")

st.subheader("Standings")
pct = lambda x: "–" if x != x else f"{x:.0%}"  # noqa: E731  (NaN -> dash)
st.dataframe(
    lb.assign(win_rate=lb["win_rate"].map(pct), roi=lb["roi"].map(pct))[["rank", "username", "balance", "net", "wagers", "wins", "losses", "win_rate", "roi", "parlays",
        "open_stakes", "biggest_win"]],
    hide_index=True, width="stretch",
    column_config={
        "rank": "#", "username": "Player",
        "balance": st.column_config.NumberColumn("Balance", format="%.2f"),
        "net": st.column_config.NumberColumn("Net +/-", format="%+.2f",
                                             help="Balance + money on open bets − starting 1,000"),
        "wagers": "Bets", "wins": "W", "losses": "L",
        "win_rate": "Win %",
        "roi": st.column_config.TextColumn("ROI", help="Profit ÷ amount staked (settled)"),
        "parlays": "Parlays",
        "open_stakes": st.column_config.NumberColumn("Open stakes", format="%.2f"),
        "biggest_win": st.column_config.NumberColumn("Biggest win", format="%.2f"),
    },
)
if user and user in set(lb["username"]):
    me = lb.set_index("username").loc[user]
    st.caption(f"You (**{user}**) are ranked **#{int(me['rank'])}** of {len(lb)}. "
               "Win % and ROI appear once bets settle.")
top = lb.head(15).iloc[::-1]
fig = go.Figure(go.Bar(y=top["username"], x=top["net"], orientation="h",
                       marker_color=[BLUE if v >= 0 else RED for v in top["net"]],
                       hovertemplate="%{y}: %{x:+,.2f}<extra></extra>"))
fig.add_vline(x=0, line_color=GRAY)
fig.update_layout(height=max(240, 30 * len(top) + 80), xaxis_title=f"Net {CURRENCY}")
st.plotly_chart(styled(fig, "Net profit by player"), width="stretch")

st.divider()
st.subheader("🏦 The House: two separate ledgers")
st.markdown("Every stake goes to the house when a bet is placed; the house pays winners and keeps losing stakes. "
            "Wagers placed with the **house edge on** and with **fair odds** are booked separately, so you can see "
            "how much the margin is worth over time. The highlighted card is the mode you're in now.")
style.house_ledgers(db.house_summary("edge"), db.house_summary("fair"), current_book(), CURRENCY)
st.write("")
tl = db.house_timeline()
if tl.empty:
    st.caption("The chart appears once wagers start settling.")
else:
    fig = go.Figure()
    for book, color, name in (("edge", "#c97a00", "House edge ON book"), ("fair", "#0a7d3b", "Fair-odds book")):
        d = tl[tl["book"] == book]
        if not d.empty:
            fig.add_scatter(x=d["settled_at"], y=d["house_profit"], mode="lines+markers", name=name,
                            line=dict(color=color, width=2, shape="hv"), marker=dict(size=7),
                            hovertemplate="%{x|%b %d %H:%M}: %{y:+,.2f}<extra>" + name + "</extra>")
    fig.add_hline(y=0, line_dash="dot", line_color=GRAY)
    fig.update_layout(yaxis_title=f"Cumulative house profit ({CURRENCY})", xaxis_title="Settled")
    st.plotly_chart(styled(fig, "House profit over time: edge vs. fair odds"), width="stretch")
    st.caption("Over many bets the edge book trends upward (the margin), while the fair book hovers around zero: "
               "it wins and loses only by luck.")
