"""AI Analyst: an LLM-written EDA report and a data-aware chatbot."""

import streamlit as st

from src import database as db
from src import eda, llm
from src.betting import VOID_STATUSES, betting_summary
from src.odds_model import make_lines
from src.ui import backtest_summary, get_backtest, get_games, get_ratings, season_label, sidebar, stream_llm

games = get_games()
user = sidebar()
ratings = get_ratings(games)

st.title("🤖 AI Analyst")
st.caption(f"Powered by Claude through the Anthropic API (model `{st.session_state.get('model', llm.DEFAULT_MODEL)}`). "
           "The AI sees only the data summaries this app computes (expand *Data sent to the model* to see "
           "them), so its answers are grounded in our analysis rather than the internet.")

if not llm.is_configured():
    st.warning("""
**AI features need an Anthropic API key.** The rest of the app works without one.

1. Get a key at https://console.anthropic.com
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


def latest_records() -> dict:
    """Each team's current W-L record from the most recent game ESPN reported."""
    cur = games[games["season"] == seasons[0]].sort_values("tipoff_utc")
    rec = {}
    for side in ("home", "away"):
        rec.update(cur.dropna(subset=[f"{side}_record"]).groupby(f"{side}_team")[f"{side}_record"].last().to_dict())
    return rec


upcoming = games[~games["completed"] & ~games["status"].isin(VOID_STATUSES)].sort_values("tipoff_utc").head(15)
slate = []
for g in upcoming.itertuples():
    ln = make_lines(ratings, g.home_team, g.away_team, g.neutral_site)
    slate.append({"game": f"{g.away_team} @ {g.home_team}", "date": str(g.game_date.date()),
                  "home_win_prob": round(ln["p_home"], 3), "home_moneyline": ln["home_ml"],
                  "away_moneyline": ln["away_ml"], "home_spread": ln["home_spread"], "total": ln["total"]})
records = latest_records()
context = {
    "current_season": season_label(seasons[0]),
    "eda_current_season": eda_insights(games, (seasons[0],)),
    "power_ratings": {t: {"net_rating": round(float(v), 2), "record": records.get(t)}
                      for t, v in ratings.ratings.items()},
    "home_court_advantage_pts": round(ratings.home_court, 2),
    "upcoming_games_with_lines": slate,
    "model_backtest": backtest_summary(get_backtest(games)),
}
if user:
    my_bets = db.load_bets(user)
    context["user_betting"] = {
        **betting_summary(my_bets), "balance": db.get_balance(user),
        "recent_bets": my_bets.head(10)[["matchup", "bet_type", "selection", "line", "american_odds",
                                         "stake", "status"]].to_dict("records"),
    }

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
                "What does the data say about 3-point shooting?",
                "Who are the 3 best teams and why?",
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
