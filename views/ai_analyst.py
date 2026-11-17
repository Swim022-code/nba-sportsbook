"""AI Analyst: an LLM-written EDA report and a data-aware chatbot."""

import pandas as pd
import streamlit as st

from src import ai_context, eda, futures, llm, style
from src import database as db
from src.betting import betting_summary
from src.ui import (
    backtest_summary,
    get_backtest,
    get_games,
    get_ratings,
    get_title_sim,
    house_edge_on,
    player_games,
    season_label,
    sidebar,
    stream_llm,
)

games = get_games()
user = sidebar()
ratings = get_ratings(games)

style.hero("AI Analyst", "Claude explains our data and answers your questions, grounded in the numbers this app computes", "🤖")
st.caption(f"Powered by Claude through the Anthropic API (model `{st.session_state.get('model', llm.DEFAULT_MODEL)}`). "
           "The AI sees only the data summaries this app computes (expand *Data sent to the model* to see "
           "them), so its answers are grounded in our analysis rather than the internet.")

if not llm.is_configured():
    st.warning("""
**AI features need an Anthropic API key.** The rest of the app works without one.

1. Get a key at https://platform.claude.com/settings/keys (add a few dollars of credit under Billing)
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste in your key
3. Restart the app (`Ctrl+C`, then `streamlit run app.py`)
""", icon="🔑")

# ---------------------------------------------------------------- grounding data
seasons = sorted(games["season"].unique(), reverse=True)


@st.cache_data(ttl=1800)
def eda_insights(_games, season_key: tuple) -> dict:
    done = _games[_games["completed"] & _games["season"].isin(season_key)]
    tg, _ = eda.treat_missing(eda.to_team_games(done))
    return eda.key_insights(done, tg)


now = pd.Timestamp.now(tz="UTC")
season = int(seasons[0])
pg = player_games()
edge = house_edge_on()
context = {
    "about_this_app": ai_context.model_description(ratings, edge),
    "data_coverage": ai_context.DATA_COVERAGE,
    "current_season": season_label(season),
    "power_ratings": ai_context.power_ratings(ratings, futures.standings(games, season)),
    "upcoming_games": ai_context.upcoming_games(games, ratings, pg, edge, now),
    "championship_futures": ai_context.futures_board(get_title_sim(games, ratings, season), edge),
    "player_leaders_min_5_games": ai_context.player_leaders(pg, season),
    "model_backtest": backtest_summary(get_backtest(games)),
    "eda_current_season": eda_insights(games, (season,)),
    "house_ledgers": {"edge_book": db.house_summary("edge"), "fair_book": db.house_summary("fair")},
}
if user:
    my_bets, my_parlays = db.load_bets(user), db.load_parlays(user)
    context["user_betting"] = ai_context.user_betting(
        betting_summary(pd.concat([my_bets, my_parlays], ignore_index=True)), db.get_balance(user), my_bets,
        my_parlays, db.load_parlay_legs(my_parlays["parlay_id"].tolist()))

tab_report, tab_chat = st.tabs(["📝 AI-written EDA report", "💬 Ask the analyst"])

# ---------------------------------------------------------------- report
with tab_report:
    st.markdown("Claude turns the statistics from **📊 Data Analysis** into a written report.")
    scope = st.radio("Data to cover", ["current", "all"], horizontal=True,
                     format_func={"current": f"Current season ({season_label(seasons[0])})",
                                  "all": "All seasons in the dataset"}.get)
    insights = eda_insights(games, (seasons[0],) if scope == "current" else tuple(seasons))
    if st.button("✍️ Generate report", type="primary", disabled=not llm.is_configured()):
        text = stream_llm(llm.write_eda_report, insights)
        if text:
            st.session_state["eda_report"] = text
    elif "eda_report" in st.session_state:
        st.markdown(st.session_state["eda_report"])
    if "eda_report" in st.session_state:
        st.download_button("⬇️ Download report (.md)", st.session_state["eda_report"], "ai_eda_report.md")
    with st.expander("Data sent to the model"):
        st.json(insights)

# ---------------------------------------------------------------- chat
with tab_chat:
    st.markdown("Ask about teams, upcoming lines, the analysis, or your own bets. Try one:")
    examples = ["Which upcoming game has the closest odds?",
                "Who is the title favorite and why?",
                "Who leads the league in scoring?",
                "How am I doing with my bets?"]
    cols = st.columns(len(examples))
    clicks = [c.button(q, key=f"ex{i}", disabled=not llm.is_configured(), width="stretch")
              for i, (c, q) in enumerate(zip(cols, examples, strict=True))]
    clicked = next((q for q, hit in zip(examples, clicks, strict=True) if hit), None)

    history = st.session_state.setdefault("chat_history", [])
    conversation = st.container()  # messages render here, above the input box
    question = st.chat_input("Ask the AI analyst...", disabled=not llm.is_configured()) or clicked
    with conversation:
        for msg in history:
            st.chat_message(msg["role"]).markdown(msg["content"])
        if question:
            st.chat_message("user").markdown(question)
            with st.chat_message("assistant"):
                answer = stream_llm(llm.chat, question, history, context)
            if answer:
                history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        if history and st.button("🗑️ Clear chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    with st.expander("Data the chatbot can see"):
        st.json(context)
