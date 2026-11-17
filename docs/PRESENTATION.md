# 15-Minute Presentation & Live Demo Plan

Each member presents the part they built. Rehearse once with a timer.

| Time | Section | Who | What to show / say |
|---|---|---|---|
| 0:00–1:00 | **Hook** | Member 1 | "We built a sportsbook where *we* are the house. Every line comes from our own model, and we test whether it's any good." Show **Home**. |
| 1:00–3:00 | **Data acquisition** | Member 1 | ESPN API: free JSON, one request per month, cached in SQLite, offline CSV fallback. Show *Data Analysis → Overview → Data source* and the JSON example in the README. |
| 3:00–6:00 | **EDA** | Member 2 | *Data quality*: **the All-Star Game story** (games with missing box scores revealed ESPN labels the All-Star Game as regular season) plus **postponed games**. *Outliers*: move the IQR slider; winsorizing barely changes the mean. *Correlations*: **FG% r = 0.47 vs. 3PT attempts r ≈ 0, so efficiency beats volume**. *Trends*: **playoff games score about 15 fewer points**. Click **🤖 Explain this with AI** once. |
| 6:00–8:30 | **The model** | Member 3 | *Teams*: ridge-regression power ratings, home court ≈ +2.3 pts. *Model evaluation*: **68% accuracy vs. 55% for always picking home**, Brier 0.208 vs. 0.25, and the **calibration plot** (key slide: 70% favorites win about 70%). |
| 8:30–12:30 | **Live demo** | Member 4 | Log in → **Sportsbook** → pick a game → **AI matchup preview** → place a spread bet → balance drops instantly → **My Bets** shows yesterday's **settled** bets and the profit chart → **AI Analyst** → *Generate report* → chatbot: "Which upcoming game has the closest odds?" then "How am I doing with my bets?" |
| 12:30–14:00 | **Engineering** | Anyone | `src/` (logic) vs `views/` (UI); 34 pytest tests incl. a fake-LLM test; GitHub Actions CI badge/run; PR workflow; graceful errors (no key, ESPN offline, postponed games). |
| 14:00–15:00 | **Lessons & next steps** | Anyone | Real data is messy; evaluate honestly (walk-forward); ground the LLM in your own numbers. Next: injuries/rest days, compare with real sportsbook lines, parlays. |

## Demo-day checklist
- [ ] **The night before:** log in as your demo user and place 3–4 bets on that night's games, so by class they've **settled** and My Bets shows a W-L record and profit chart
- [ ] Start the app on the presenting laptop **before** class: `streamlit run app.py` (first load downloads data)
- [ ] `.streamlit/secrets.toml` has a working key; click one **AI matchup preview** before presenting
- [ ] Backup: take screenshots of AI output in case the Wi-Fi fails (the rest of the app works offline)
- [ ] Browser zoom 110–125% so the room can read it; close other tabs

## Likely questions & answers
- **Why not use real sportsbook odds?** The free API doesn't have them, and building our own odds model was the data-science challenge. Comparing against real lines is future work.
- **Why ridge regression?** Early in the season teams have only about 13 games. Ridge pulls noisy ratings toward average, and recency weighting tracks roster changes.
- **How do you know it isn't overfit?** The walk-forward back-test only predicts games *after* each training cutoff, refitting weekly.
- **Does the LLM make up stats?** Every prompt contains our computed JSON, and the system prompt says to use only those numbers. The app shows exactly what's sent ("Data sent to the model").
- **What if the API is down?** ESPN down → cached data, then the CSV snapshot. Claude down or no key → a clear message, and the rest of the app keeps working.
