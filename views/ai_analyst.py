"""AI Analyst: an LLM-written EDA report and a data-aware chatbot."""

import streamlit as st

from src import database as db
from src import eda, llm
from src.betting import betting_summary
from src.odds_model import make_lines
from src.ui import get_games, get_ratings, sidebar, stream_llm

user = sidebar()
games = get_games()
ratings = get_ratings(games)

st.title("🤖 AI Analyst")
st.caption(f"Powered by Claude via the Anthropic API (model: `{st.session_state.get('model', llm.DEFAULT_MODEL)}`). "
           "The AI only sees the data summaries listed below, not the internet.")

if not llm.is_configured():
    st.warning("""
**AI features need an Anthropic API key.**

1. Get a key at https://console.anthropic.com
2. Create `.streamlit/secrets.toml` containing:  `ANTHROPIC_API_KEY = "sk-ant-..."`
3. Restart the app.
""", icon="🔑")

# Build the context the LLM is grounded in.
current_season = int(games["season"].max())
done = games[games["completed"] & (games["season"] == current_season)]
tg, _ = eda.treat_missing(eda.to_team_games(done))
insights = eda.key_insights(done, tg)
upcoming = games[~games["completed"]].sort_values("tipoff_utc").head(15)
slate = []
for g in upcoming.itertuples():
    ln = make_lines(ratings, g.home_team, g.away_team, g.neutral_site)
    slate.append({"game": f"{g.away_team} @ {g.home_team}", "date": str(g.game_date.date()),
                  "home_win_prob": round(ln["p_home"], 3), "home_ml": ln["home_ml"], "away_ml": ln["away_ml"],
                  "home_spread": ln["home_spread"], "total": ln["total"]})
context = {
    "season": f"{current_season - 1}-{str(current_season)[-2:]} (current season only)",
    "eda": insights,
    "power_ratings_top_to_bottom": ratings.ratings.round(2).to_dict(),
    "home_court_advantage_pts": round(ratings.home_court, 2),
    "upcoming_games_with_lines": slate,
    "model_backtest": st.session_state.get("backtest_summary", "visit the Data Analysis page to compute"),
}
if user:
    my_bets = db.load_bets(user)
    context["user_betting"] = {**betting_summary(my_bets), "balance": db.get_balance(user),
                               "recent_bets": my_bets.head(10)[["matchup", "bet_type", "selection", "line",
                                                                "american_odds", "stake", "status"]].to_dict("records")}

tab_report, tab_chat = st.tabs(["📝 AI-generated EDA report", "💬 Ask the analyst"])

with tab_report:
    st.markdown("Claude turns the statistics computed on the Data Analysis page into a written report.")
    if st.button("Generate report", type="primary", disabled=not llm.is_configured()):
        text = stream_llm(llm.write_eda_report, insights)
        if text:
            st.session_state["eda_report"] = text
            st.download_button("Download report (.md)", text, "ai_eda_report.md")
    elif "eda_report" in st.session_state:
        st.markdown(st.session_state["eda_report"])
    with st.expander("Data sent to the model"):
        st.json(insights)

with tab_chat:
    st.markdown("Ask about teams, tonight's lines, the analysis, or your own bets. Examples:")
    examples = ["Which upcoming game has the closest odds?",
                "Explain what the correlation analysis says about 3-point shooting.",
                "Who are the 3 best teams and why?",
                "How am I doing with my bets?"]
    cols = st.columns(len(examples))
    clicked = next((q for c, q in zip(cols, examples) if c.button(q, disabled=not llm.is_configured())), None)

    history = st.session_state.setdefault("chat_history", [])
    for msg in history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    question = st.chat_input("Ask the AI analyst...", disabled=not llm.is_configured()) or clicked
    if question:
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"):
            answer = stream_llm(llm.chat, question, history, context)
        if answer:
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    if history and st.button("Clear chat"):
        st.session_state["chat_history"] = []
        st.rerun()

    with st.expander("Data the chatbot can see"):
        st.json(context)
