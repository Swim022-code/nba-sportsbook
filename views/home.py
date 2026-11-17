"""Home page: overview, how-to guide and top power ratings."""

import streamlit as st

from src import database as db
from src.ui import CURRENCY, get_games, get_ratings, sidebar


user = sidebar()
games = get_games()
ratings = get_ratings(games)

st.title("🏀 NBA Mock Sportsbook")
st.markdown(
    f"Bet on real upcoming NBA games with **fake money** ({CURRENCY}). "
    "Lines are set by our own statistical model, built from live ESPN data, "
    "and an AI analyst (Claude) explains the numbers."
)

completed = games[games["completed"]]
upcoming = games[~games["completed"]]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games in dataset", f"{len(games):,}")
c2.metric("Completed games", f"{len(completed):,}")
c3.metric("Upcoming games", f"{len(upcoming):,}")
c4.metric("Home-court edge (model)", f"{ratings.home_court:+.1f} pts")

st.subheader("How to use this app")
st.markdown(f"""
1. **Log in** in the sidebar with any username. New accounts get **{db.STARTING_BALANCE:,.0f} {CURRENCY}**.
2. **🏀 Sportsbook**: browse upcoming games, see moneyline / spread / total odds, and place bets.
   Ask the AI for a matchup preview before you bet.
3. **🎟️ My Bets**: track open bets. Bets settle automatically once ESPN reports a final score.
4. **📊 Data Analysis**: the exploratory data analysis behind the odds: distributions,
   outliers, missing data, correlations, team ratings and a back-test of the model.
5. **🤖 AI Analyst**: an AI-written EDA report plus a chatbot that answers questions about the data.
""")

with st.expander("📖 Betting basics"):
    st.markdown("""
- **Moneyline**: pick the winner. `-150` means risk 150 to win 100; `+130` means risk 100 to win 130.
- **Spread**: the favorite must win by more than the line. `BOS -5.5` wins only if Boston wins by 6+.
- **Total (Over/Under)**: bet whether the combined score goes over or under the line.
- A result exactly on the line is a **push**, and your stake is refunded.
- The house builds in a small margin (the "vig"), so the implied probabilities add up to more than 100%.
""")

st.subheader("Top 10 teams by model power rating")
st.caption("Points better than an average team on a neutral court, estimated from game results.")
top = ratings.ratings.head(10).rename("Net rating").to_frame()
top["Off. (pts scored)"] = ratings.offense.reindex(top.index)
top["Def. (pts allowed)"] = ratings.defense.reindex(top.index)
st.dataframe(top.round(1), width="stretch")

if not user:
    st.info("👈 Log in from the sidebar to start betting.")
