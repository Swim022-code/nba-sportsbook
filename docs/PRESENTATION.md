# 15-Minute Presentation & Live Demo Plan

Split the speaking parts between team members; each person presents what they built.

| Time | Section | Who | What to show / say |
|---|---|---|---|
| 0:00–1:00 | **Hook** | Member 1 | "We built a sportsbook. The house sets every line with our own model, and we test whether it's any good." Show the Home page. |
| 1:00–3:00 | **Data acquisition** | Member 1 | ESPN API endpoint, JSON structure (README section 4), monthly fetching plus SQLite caching, and the offline CSV fallback. Show the *Data Analysis → Overview → Data source* table. |
| 3:00–6:00 | **EDA** | Member 2 | *Data quality* tab: the **All-Star Game story** (missing box scores revealed ESPN mislabels the All-Star Game as regular season). Outliers: move the IQR slider and show winsorizing. *Correlations*: FG% r = 0.47 with winning vs. 3PT attempts ≈ 0, so **efficiency beats volume**. Home teams win about 55%. |
| 6:00–8:30 | **The model** | Member 3 | *Teams* tab: ridge-regression power ratings. *Model evaluation*: the walk-forward back-test, **68% accuracy vs. 55% for always picking home**, and the **calibration plot** (the key slide: 70% favorites win about 70% of the time). |
| 8:30–12:30 | **Live demo** | Member 4 | Log in → Sportsbook → pick a game → **AI matchup preview** → place a spread bet → My Bets shows it pending and the balance drops → AI Analyst → **Generate report** → ask the chatbot "Which upcoming game has the closest odds?" |
| 12:30–14:00 | **Engineering & collaboration** | Anyone | Module structure, pytest suite (`pytest -q`, 21 tests), GitHub branches/PRs, error handling (try it with no API key: the app still works). |
| 14:00–15:00 | **Lessons & future work** | Anyone | Real data is messy, honest evaluation matters, ground the LLM in your own data. Next: injuries/rest days, comparing to real lines. |

## Demo-day checklist
- [ ] `streamlit run app.py` runs on the presenting laptop **before** class (first load downloads data)
- [ ] `.streamlit/secrets.toml` has a working `ANTHROPIC_API_KEY`; run one AI preview beforehand
- [ ] Create a user and place 2–3 bets the day before, so some have **settled** and My Bets shows a W-L record and profit chart
- [ ] Backup: if the Wi-Fi fails, the app falls back to `data/games_snapshot.csv`. AI features need internet, so keep screenshots of AI output ready
- [ ] Browser zoom at 110–125% so the room can read it

## Likely questions
- **Why not use real sportsbook odds?** The free API doesn't include them. Building our own odds model was the data-science challenge.
- **Why ridge regression?** Early in the season teams have only about 10 games. Ridge shrinks noisy ratings toward average, and recency weighting tracks roster changes.
- **How do you know the model isn't overfit?** The walk-forward back-test only ever predicts games after the training cutoff.
- **Does the LLM make up stats?** Each prompt contains our computed JSON, and the system prompt says to use only those numbers. The app shows the exact data sent ("Data sent to the model").
