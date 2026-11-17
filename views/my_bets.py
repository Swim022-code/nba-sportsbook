"""My Bets: open and settled bets, profit/loss chart, leaderboard."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import database as db
from src import style
from src.betting import betting_summary
from src.data_fetcher import EASTERN
from src.ui import CURRENCY, fmt_line, fmt_odds, get_games, house_edge_on, money, sidebar

games = get_games()  # also settles any finished bets
user = sidebar()

style.hero("My Bets", "Your open and settled bets, results and profit", "🎟️", style.edge_badge(house_edge_on()))
if not user:
    st.info("👈 Log in from the sidebar to see your bets.")
    st.stop()

bets = db.load_bets(user)
s = betting_summary(bets)
balance = db.get_balance(user)

c = st.columns(5)
c[0].metric("Balance", money(balance))
c[1].metric("Open bets", s["pending"])
c[2].metric("Record (W-L-P)", f"{s['won']}-{s['lost']}-{s['push']}",
            help="Wins - losses - pushes. Voided bets (postponed games) are refunded and not counted.")
c[3].metric("Profit (settled)", money(s["profit"]))
c[4].metric("ROI", f"{s['roi']:.1%}")

if bets.empty:
    st.info("No bets yet. Head to **🏀 Sportsbook** to place one.")
else:
    def describe(row):
        if row.bet_type == "moneyline":
            return f"{row.selection} ML {fmt_odds(row.american_odds)}"
        if row.bet_type == "spread":
            return f"{row.selection} {fmt_line(row.line)} ({fmt_odds(row.american_odds)})"
        return f"{row.selection.title()} {row.line:g} ({fmt_odds(row.american_odds)})"

    bets["bet"] = bets.apply(describe, axis=1)
    scores = games.set_index("game_id")[["home_team", "home_score", "away_team", "away_score"]]
    bets = bets.join(scores, on="game_id")
    bets["final"] = bets.apply(lambda r: f"{r.away_team} {r.away_score:.0f} - {r.home_team} {r.home_score:.0f}"
                               if pd.notna(r.home_score) else "", axis=1)

    status_filter = st.segmented_control("Show", ["all", "pending", "won", "lost", "push", "void"], default="all")
    view = bets if status_filter in (None, "all") else bets[bets["status"] == status_filter]
    icons = {"pending": "⏳", "won": "✅", "lost": "❌", "push": "↩️", "void": "🚫"}
    view = view.assign(status=view["status"].map(lambda x: f"{icons[x]} {x}"))
    view = view.assign(placed=pd.to_datetime(view["placed_at"], utc=True).dt.tz_convert(EASTERN).dt.strftime("%b %d %I:%M %p"))
    num = st.column_config.NumberColumn
    st.dataframe(
        view[["bet_id", "placed", "matchup", "bet", "stake", "potential_payout", "status", "payout", "final"]],
        hide_index=True, width="stretch",
        column_config={"bet_id": num("Bet #"), "placed": "Placed (ET)", "matchup": "Game", "bet": "Bet",
                       "stake": num("Stake", format="%.2f"),
                       "potential_payout": num("Returns if won", format="%.2f",
                                               help="Stake + profit paid out if the bet wins"),
                       "status": "Status", "payout": num("Paid out", format="%.2f"), "final": "Final score"},
    )

    settled = bets[bets["status"] != "pending"].sort_values(["settled_at", "bet_id"])
    if not settled.empty:
        settled = settled.assign(pnl=settled["payout"] - settled["stake"])
        settled["cumulative_pnl"] = settled["pnl"].cumsum()
        settled["n"] = range(1, len(settled) + 1)
        fig = px.line(settled, x="n", y="cumulative_pnl", markers=True, hover_data=["bet", "status", "pnl"],
                      title="Cumulative profit / loss", color_discrete_sequence=["#2a78d6"],
                      labels={"cumulative_pnl": f"Profit ({CURRENCY})", "n": "Settled bets", "pnl": "This bet"})
        fig.update_traces(line_width=2, marker_size=9)
        fig.update_xaxes(dtick=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#898781")
        st.plotly_chart(fig, width="stretch")

if house_edge_on():  # the house's books are shown only in house-edge mode
    style.house_panel(db.house_summary(), CURRENCY)
    st.write("")

st.subheader("🏆 Leaderboard")
st.dataframe(db.leaderboard(), hide_index=True, width="stretch",
             column_config={"username": "Player", "balance": st.column_config.NumberColumn("Balance", format="%.2f"),
                            "bets": "Bets", "wins": "Wins",
                            "at_risk": st.column_config.NumberColumn("Open stakes", format="%.2f")})

with st.expander("⚠️ Reset my account"):
    st.write(f"Deletes all your bets and resets your balance to {db.STARTING_BALANCE:,.0f} {CURRENCY}.")
    if st.button("Reset account", type="secondary"):
        db.reset_user(user)
        st.success("Account reset.")
        st.rerun()
