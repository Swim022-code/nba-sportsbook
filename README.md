# 🏀 NBA Mock Sportsbook

A Streamlit web app where users bet **fake currency ("SimBucks")** on **real upcoming NBA games**.
The betting lines are set by **our own statistical model** built from live ESPN data, the app
includes a full **exploratory data analysis (EDA)**, and an **LLM (Claude, via the Anthropic API)**
writes reports, previews games and answers questions about the data.

> Masters of Data Science, Programming for Data Science, final project.
> No real money is involved at any point.

---

## 1. Project purpose & overview

Sportsbooks are a real-world data-science product: they turn historical data into probabilities,
then into prices. This project rebuilds that pipeline end to end:

| Step | What happens | Where |
|---|---|---|
| **Acquire** | Download every NBA game from last season and this season from the ESPN API | `src/data_fetcher.py`, `src/data_service.py` |
| **Store** | Cache games, users and bets in SQLite | `src/database.py` |
| **Analyze (EDA)** | Descriptive stats, missing data, outliers, correlations | `src/eda.py`, `views/data_analysis.py` |
| **Model** | Team power ratings (weighted ridge regression) → win probabilities → odds | `src/odds_model.py` |
| **Evaluate** | Walk-forward back-test: accuracy, Brier score, calibration | `src/odds_model.py::backtest` |
| **Bet** | Place moneyline / spread / total bets; auto-settle when games end | `src/betting.py`, `views/sportsbook.py` |
| **LLM** | AI EDA report, AI matchup previews, AI analyst chatbot | `src/llm.py`, `views/ai_analyst.py` |

### App pages
- **🏠 Home**: overview, how-to guide, top power ratings
- **🏀 Sportsbook**: upcoming games with model odds, a bet slip, and an AI matchup preview
- **🎟️ My Bets**: open/settled bets, record, ROI, profit chart, leaderboard
- **📊 Data Analysis**: interactive EDA (filters by season, game type and team) and model evaluation
- **🤖 AI Analyst**: an LLM-written EDA report and a chatbot grounded in the app's data

---

## 2. Setup & installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/<your-team>/nba-sportsbook.git
cd nba-sportsbook
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Add your Anthropic API key (for the AI features)
1. Create a key at <https://console.anthropic.com>.
2. Copy the example secrets file and paste in your key:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   (or `export ANTHROPIC_API_KEY=sk-ant-...`)

`secrets.toml` is git-ignored, so your key is never committed. **Without a key the app still works**;
only the AI features are disabled, and they show setup instructions instead.

---

## 3. How to run the Streamlit app

```bash
streamlit run app.py
```
Then open <http://localhost:8501>. On first launch the app downloads about 1,800 games from ESPN
(10–20 seconds) and caches them in `data/sportsbook.db`. After that it only re-fetches the current
month and the upcoming week, at most every 30 minutes, or when you click **🔄 Refresh data**.

Run the tests:
```bash
pytest -q
```

### Deploying to Streamlit Community Cloud (optional)
Push to GitHub → <https://share.streamlit.io> → *New app* → pick the repo, set the main file to `app.py`,
and add `ANTHROPIC_API_KEY` under *Advanced settings → Secrets*.

---

## 4. API & data source details

### ESPN NBA scoreboard API (data)
| | |
|---|---|
| Endpoint | `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard` |
| Auth | None (free, public, unofficial) |
| Format | **JSON** |
| Queries used | `?dates=YYYYMM&limit=500` (a whole month of games), `?dates=YYYYMMDD` (one day) |
| Coverage | 2025-26 season (incl. play-in & playoffs) + 2026-27 season to date + next 7 days of schedule |

Response structure (trimmed):
```json
{"events": [{
  "id": "401909993",
  "date": "2026-11-16T03:00Z",
  "season": {"year": 2027, "slug": "regular-season"},
  "status": {"type": {"name": "STATUS_FINAL", "completed": true}},
  "competitions": [{
    "neutralSite": false, "attendance": 18236, "venue": {"fullName": "..."},
    "competitors": [{
      "homeAway": "home", "score": "103", "winner": false,
      "team": {"abbreviation": "SAC", "displayName": "Sacramento Kings"},
      "statistics": [{"name": "rebounds", "displayValue": "37"}, {"name": "fieldGoalPct", "displayValue": "45.1"}],
      "records": [{"type": "total", "summary": "1-12"}]
    }, {"homeAway": "away", "...": "..."}]
  }]
}]}
```
Each event is flattened into **one row per game** with `home_*` / `away_*` columns: score, points, rebounds,
assists, FG/3PT/FT made/attempted/%, and record. For the analysis it is reshaped to
**one row per team per game**. Preseason and exhibition games are excluded.

**Offline fallback:** `data/games_snapshot.csv` holds a snapshot of the data. If ESPN can't be
reached on first launch, the app loads it so the demo still works.

### Anthropic Claude API (LLM)
- SDK: `anthropic` (official Python SDK), streaming responses
- Default model: `claude-opus-5-5` (can be changed in the sidebar)
- The LLM is **grounded**: each prompt includes JSON computed by the app (EDA results, ratings, odds,
  the user's bets), and the system prompt tells it to cite only those numbers
- Used for: (1) a written EDA report, (2) game previews comparing the model's probability with the
  odds, (3) a chatbot for questions about the data
- Error handling: missing or invalid key, rate limits, network errors and refusals each show a
  clear message instead of crashing

---

## 5. Methodology summary

**Power ratings.** Fit `margin = HCA·home + rating_home − rating_away` with weighted ridge regression
(λ = 2). Weights decay with a 45-day half-life, and last season's games count at 35%. A second
regression splits each rating into offense (points scored) and defense (points allowed), which is
used to set totals.

**Odds.** `P(home win) = Φ(predicted margin / σ)`, where σ is the standard deviation of the model's
errors. Moneylines add a 4.5% house margin; spreads and totals are priced at -110.

**Evaluation.** A walk-forward back-test refits every week using only earlier games. It picks winners
about **68%** of the time (always picking the home team gets about 55%), with a Brier score of about
**0.21** and a calibration curve that tracks the diagonal.

**Settlement.** Moneyline, spread and total bets are graded from ESPN final scores. Results exactly
on the line are pushes and the stake is refunded.

---

## 6. Project structure
```
nba-sportsbook/
├── app.py                  # entry point + page navigation
├── views/                  # one file per page
│   ├── home.py  sportsbook.py  my_bets.py  data_analysis.py  ai_analyst.py
├── src/
│   ├── data_fetcher.py     # ESPN API calls + JSON parsing
│   ├── data_service.py     # caching / refresh logic, offline snapshot
│   ├── database.py         # SQLite schema + queries
│   ├── odds_model.py       # ratings, odds math, back-test
│   ├── betting.py          # place / grade / settle bets
│   ├── eda.py              # EDA functions (pure pandas, unit-tested)
│   ├── charts.py           # Plotly chart builders
│   ├── llm.py              # Claude API integration
│   └── ui.py               # shared Streamlit helpers (sidebar, caching)
├── tests/test_core.py      # pytest unit tests
├── data/games_snapshot.csv # offline data snapshot
├── docs/                   # report, presentation outline, screenshots
└── requirements.txt
```

## 7. Team & collaboration
| Member | Main responsibilities |
|---|---|
| _Name 1_ | Data acquisition, database |
| _Name 2_ | EDA & visualizations |
| _Name 3_ | Odds model & betting engine |
| _Name 4_ | LLM integration & Streamlit UI |

We worked on feature branches, opened pull requests into `main`, and reviewed each other's code.
