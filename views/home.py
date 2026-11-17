"""Home page: overview, how-to guide, next games and top power ratings."""

import pandas as pd
import streamlit as st

from src import database as db
from src import style
from src.betting import VOID_STATUSES
from src.data_fetcher import EASTERN
from src.odds_model import make_lines
from src.ui import CURRENCY, fmt_line, fmt_odds, get_games, get_ratings, house_edge_on, sidebar

games = get_games()
user = sidebar()
ratings = get_ratings(games)

style.hero("NBA Mock Sportsbook",
           f"Bet on real upcoming NBA games with fake money ({CURRENCY}). Odds from our own model on live "
           "ESPN data, explained by an AI analyst.", "🏀", style.edge_badge(house_edge_on()))
if not user:
    st.info("👈 **Start here:** pick a username in the sidebar to get your free "
            f"{db.STARTING_BALANCE:,.0f} {CURRENCY}.", icon="🎟️")

now = pd.Timestamp.now(tz="UTC")
completed = games[games["completed"]]
upcoming = games[~games["completed"] & ~games["status"].isin(VOID_STATUSES) & (games["tipoff_utc"] > now)]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games analyzed", f"{len(completed):,}")
c2.metric("Open for betting", f"{len(upcoming):,}")
c3.metric("Teams rated", len(ratings.ratings))
c4.metric("Home-court edge", f"{ratings.home_court:+.1f} pts")

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("⏭️ Next games")
    rows = []
    for g in upcoming.sort_values("tipoff_utc").head(6).itertuples():
        ln = make_lines(ratings, g.home_team, g.away_team, g.neutral_site, house_edge=house_edge_on())
        rows.append({"Tip-off (ET)": g.tipoff_utc.tz_convert(EASTERN).strftime("%a %I:%M %p"),
                     " ": style.logo_url(g.away_team), "Away": f"{g.away_team} {fmt_odds(ln['away_ml'])}",
                     "  ": style.logo_url(g.home_team), "Home": f"{g.home_team} {fmt_odds(ln['home_ml'])}",
                     "Spread": f"{g.home_team} {fmt_line(ln['home_spread'])}",
                     "Total": f"{ln['total']:g}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", row_height=40,
                     column_config={" ": st.column_config.ImageColumn(" ", width="small"),
                                    "  ": st.column_config.ImageColumn(" ", width="small")})
        st.caption("Away/Home show each team's moneyline.")
        st.page_link("views/sportsbook.py", label="Go to the Sportsbook to bet →", icon="🏀")
    else:
        st.caption("No games scheduled in the next week.")

with right:
    st.subheader("🏆 Top 5 power ratings")
    top = ratings.ratings.head(5).rename("Net rating").to_frame()
    top["Off."] = ratings.offense.reindex(top.index)
    top["Def."] = ratings.defense.reindex(top.index)
    top = top.round(1).reset_index(names="Team")
    top.insert(0, "Logo", top["Team"].map(style.logo_url))
    st.dataframe(top, hide_index=True, width="stretch", row_height=40,
                 column_config={"Logo": st.column_config.ImageColumn(" ", width="small")})
    st.caption("Points better than an average team on a neutral court. Off. = points scored, "
               "Def. = points allowed vs. an average opponent.")

st.subheader("How to use this app")
st.markdown(f"""
1. **Log in** in the sidebar with any username. New accounts get **{db.STARTING_BALANCE:,.0f} {CURRENCY}**.
2. **🏀 Sportsbook**: pick a game day, compare moneyline / spread / total odds, ask the AI for a preview, place a bet.
3. **🎟️ My Bets**: track your bets. They settle automatically when ESPN reports the final score.
4. **📊 Data Analysis**: the exploratory data analysis behind the odds, plus an honest test of the model.
5. **🤖 AI Analyst**: an AI-written report on the data, and a chatbot for questions.
""")

b1, b2 = st.columns(2)
with b1.expander("📖 Betting basics"):
    st.markdown("""
- **Moneyline**: pick the winner. `-150` = risk 150 to win 100. `+130` = risk 100 to win 130.
- **Spread**: the favorite must win by more than the line. `BOS -5.5` wins only if Boston wins by 6+.
- **Total (Over/Under)**: will the combined score be over or under the line?
- Landing exactly on the line is a **push** (stake refunded). Postponed games are **void** (stake refunded).
- The house adds a small margin (the "vig"), so implied probabilities add up to more than 100%.
""")
with b2.expander("⚙️ How it works (the data pipeline)"):
    st.markdown("""
1. **Acquire**: ESPN's public NBA API (JSON), monthly requests, cached in SQLite.
2. **Clean**: drop preseason and All-Star games; handle missing data and outliers.
3. **Analyze**: descriptive stats, distributions, correlations, trends.
4. **Model**: ridge-regression power ratings → win probability → odds with a house margin.
5. **Evaluate**: walk-forward back-test (accuracy, Brier score, calibration).
6. **Explain**: Claude writes reports and answers questions using only our computed numbers.
""")
