"""My Bets: open and settled bets, profit/loss chart, leaderboard."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import database as db
from src.betting import betting_summary
from src.ui import CURRENCY, fmt_line, fmt_odds, get_games, money, sidebar

user = sidebar()
games = get_games()  # also settles any finished bets

st.title("🎟️ My Bets")
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
    st.dataframe(
        view[["bet_id", "placed_at", "matchup", "bet", "stake", "potential_payout", "status", "payout", "final"]],
        hide_index=True, width="stretch",
        column_config={"stake": st.column_config.NumberColumn(format="%.2f"),
                       "potential_payout": st.column_config.NumberColumn("to win (total)", format="%.2f"),
                       "payout": st.column_config.NumberColumn(format="%.2f")},
    )

    settled = bets[bets["status"] != "pending"].sort_values("settled_at")
    if not settled.empty:
        settled = settled.assign(pnl=settled["payout"] - settled["stake"])
        settled["cumulative_pnl"] = settled["pnl"].cumsum()
        fig = px.line(settled, x="settled_at", y="cumulative_pnl", markers=True,
                      title="Cumulative profit / loss", labels={"cumulative_pnl": CURRENCY, "settled_at": ""})
        fig.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig, width="stretch")

st.subheader("🏆 Leaderboard")
st.dataframe(db.leaderboard(), hide_index=True, width="stretch",
             column_config={"balance": st.column_config.NumberColumn(format="%.2f"),
                            "at_risk": st.column_config.NumberColumn("open stakes", format="%.2f")})

with st.expander("⚠️ Reset my account"):
    st.write(f"Deletes all your bets and resets your balance to {db.STARTING_BALANCE:,.0f} {CURRENCY}.")
    if st.button("Reset account", type="secondary"):
        db.reset_user(user)
        st.success("Account reset.")
        st.rerun()
