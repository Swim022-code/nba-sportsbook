"""My Bets: singles, props, futures and parlays, with results and profit."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import database as db
from src import style
from src.betting import betting_summary
from src.data_fetcher import EASTERN
from src.ui import CURRENCY, current_book, describe_wager, get_games, house_edge_on, money, sidebar

games = get_games()  # also settles any finished wagers
user = sidebar()

style.hero("My Bets", "Your singles, props, futures and parlays: results and profit", "🎟️",
           style.edge_badge(house_edge_on()))
if not user:
    st.info("👈 Log in from the sidebar to see your bets.")
    st.stop()

bets = db.load_bets(user)
parlays = db.load_parlays(user)
wagers = pd.concat([bets.assign(kind="single"), parlays.assign(kind="parlay")], ignore_index=True)
s = betting_summary(wagers)

c = st.columns(5)
c[0].metric("Balance", money(db.get_balance(user)))
c[1].metric("Open wagers", s["pending"])
c[2].metric("Record (W-L-P)", f"{s['won']}-{s['lost']}-{s['push']}",
            help="Wins - losses - pushes. Voided bets (postponed games, players who sat) are refunded and not counted.")
c[3].metric("Profit (settled)", f"{s['profit']:+,.2f}")
c[4].metric("ROI", f"{s['roi']:.1%}")

ICONS = {"pending": "⏳", "won": "✅", "lost": "❌", "push": "↩️", "void": "🚫"}
BOOK = {"edge": "🏦 edge", "fair": "⚖️ fair"}


def placed_et(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col, utc=True, format="ISO8601").dt.tz_convert(EASTERN).dt.strftime("%b %d %I:%M %p")


status_filter = st.segmented_control("Show", ["all", "pending", "won", "lost", "push", "void"], default="all")


def keep(df: pd.DataFrame) -> pd.DataFrame:
    return df if status_filter in (None, "all") or df.empty else df[df["status"] == status_filter]


tab_s, tab_p = st.tabs([f"Single bets ({len(bets)})", f"Parlays ({len(parlays)})"])
num = st.column_config.NumberColumn

with tab_s:
    view = keep(bets)
    if view.empty:
        st.info("No single bets here yet. Head to **🏀 Sportsbook**, **🏆 Futures** or the player props.")
    else:
        scores = games.set_index("game_id")[["home_team", "home_score", "away_team", "away_score"]]
        view = view.join(scores, on="game_id")
        view = view.assign(
            bet=view.apply(lambda r: describe_wager(r.to_dict()), axis=1),
            type=view["bet_type"].str.title(), placed=placed_et(view["placed_at"]),
            status=view["status"].map(lambda x: f"{ICONS[x]} {x}"), book=view["book"].map(BOOK),
            final=view.apply(lambda r: f"{r.away_team} {r.away_score:.0f} - {r.home_team} {r.home_score:.0f}"
                             if pd.notna(r.home_score) else "", axis=1))
        st.dataframe(
            view[["bet_id", "placed", "type", "matchup", "bet", "stake", "potential_payout", "status", "payout",
                  "final", "book"]],
            hide_index=True, width="stretch",
            column_config={"bet_id": num("Bet #"), "placed": "Placed (ET)", "type": "Type", "matchup": "Game / market",
                           "bet": "Bet", "stake": num("Stake", format="%.2f"),
                           "potential_payout": num("Returns if won", format="%.2f"), "status": "Status",
                           "payout": num("Paid out", format="%.2f"), "final": "Final score", "book": "Odds mode"},
        )

with tab_p:
    view = keep(parlays)
    if view.empty:
        st.info("No parlays here yet. On the Sportsbook, use **➕ Add to parlay** on 2+ bets, then place it "
                "from the slip in the sidebar.")
    else:
        legs = db.load_parlay_legs(view["parlay_id"].tolist())
        for p in view.to_dict("records"):
            plegs = legs[legs["parlay_id"] == p["parlay_id"]]
            dec = p["decimal_odds"]
            american = round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))
            head = (f"{ICONS[p['status']]} Parlay #{p['parlay_id']} · {p['n_legs']} legs · +{american} · "
                    f"stake {money(p['stake'])} → {money(p['potential_payout'])} · {p['status']}")
            with st.expander(head):
                st.dataframe(
                    plegs.assign(bet=plegs.apply(lambda r: describe_wager(r.to_dict()), axis=1),
                                 result=plegs["result"].map(lambda x: f"{ICONS.get(x, '⏳')} {x}")
                                 )[["matchup", "bet", "result"]],
                    hide_index=True, width="stretch",
                    column_config={"matchup": "Game", "bet": "Leg", "result": "Result"})
                st.caption(f"Placed {placed_et(pd.Series([p['placed_at']])).iloc[0]} ET · {BOOK[p['book']]} odds · "
                           f"paid out {money(p['payout'])}. Pushed or voided legs drop out of the odds.")

settled = wagers[wagers["status"] != "pending"].sort_values(["settled_at"])
if not settled.empty:
    settled = settled.assign(pnl=settled["payout"] - settled["stake"])
    settled["cumulative_pnl"] = settled["pnl"].cumsum()
    settled["n"] = range(1, len(settled) + 1)
    fig = px.line(settled, x="n", y="cumulative_pnl", markers=True, hover_data=["kind", "status", "pnl"],
                  title="Cumulative profit / loss", color_discrete_sequence=["#2a78d6"],
                  labels={"cumulative_pnl": f"Profit ({CURRENCY})", "n": "Settled wagers", "pnl": "This wager"})
    fig.update_traces(line_width=2, marker_size=9)
    fig.update_xaxes(dtick=1 if len(settled) < 25 else None)
    fig.add_hline(y=0, line_dash="dot", line_color="#898781")
    st.plotly_chart(fig, width="stretch")

book = current_book()
st.subheader("🏦 The House" + (" (edge book)" if book == "edge" else " (fair-odds book)"))
h = db.house_summary(book)
st.markdown(f"In the current odds mode the house has taken **{money(h['handle'])}** {CURRENCY} on settled wagers, "
            f"paid back **{money(h['paid_out'])}**, and kept **{h['house_profit']:+,.2f}** ({h['hold_pct']:.1%} hold). "
            "See **🏆 Leaderboard** for both ledgers side by side.")

with st.expander("⚠️ Reset my account"):
    st.write(f"Deletes all your bets and parlays and resets your balance to {db.STARTING_BALANCE:,.0f} {CURRENCY}.")
    if st.button("Reset account", type="secondary"):
        db.reset_user(user)
        st.success("Account reset.")
        st.rerun()
