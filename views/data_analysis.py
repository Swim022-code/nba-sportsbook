"""Exploratory Data Analysis of NBA game data, plus model evaluation."""

import pandas as pd
import streamlit as st

from src import charts, eda
from src.data_fetcher import SCOREBOARD_URL
from src.odds_model import backtest
from src.ui import get_games, get_ratings, sidebar

sidebar()
games = get_games()
ratings = get_ratings(games)


@st.cache_data(ttl=3600, show_spinner="Back-testing the odds model...")
def run_backtest(completed: pd.DataFrame) -> pd.DataFrame:
    return backtest(completed)


st.title("📊 Exploratory Data Analysis")

# ---------------------------------------------------------------- filters
season_names = {s: f"{s - 1}-{str(s)[-2:]}" for s in sorted(games["season"].unique())}
f1, f2, f3 = st.columns(3)
seasons = f1.multiselect("Season", list(season_names), default=list(season_names),
                         format_func=season_names.get)
types = f2.multiselect("Game type", ["regular-season", "play-in-season", "post-season"],
                       default=["regular-season", "play-in-season", "post-season"])
all_teams = sorted(set(games["home_team"].dropna()) | set(games["away_team"].dropna()))
teams = f3.multiselect("Teams (empty = all)", all_teams)

g = games[games["season"].isin(seasons) & games["season_type"].isin(types)]
if teams:
    g = g[g["home_team"].isin(teams) | g["away_team"].isin(teams)]
done = g[g["completed"]]
if len(done) < 20:
    st.warning("Not enough completed games for these filters. Widen your selection.")
    st.stop()

raw_tg = eda.to_team_games(done)
if teams:
    raw_tg = raw_tg[raw_tg["team"].isin(teams)]
tg, treatment = eda.treat_missing(raw_tg)
insights = eda.key_insights(done, tg)
st.session_state["eda_insights"] = insights  # shared with the AI Analyst page

tabs = st.tabs(["📋 Overview", "🧹 Data quality", "📈 Distributions", "🔗 Correlations",
                "🏀 Teams", "🎯 Model evaluation"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Completed games", f"{insights['games_analyzed']:,}")
    c[1].metric("Avg game total", f"{insights['avg_total_points']} pts")
    c[2].metric("Home win %", f"{insights['home_win_pct']:.1%}")
    c[3].metric("Avg winning margin", f"{insights['avg_abs_margin']} pts")

    st.subheader("Key insights")
    corr = insights["correlation_with_winning"]
    top_stat = max(corr, key=corr.get)
    st.markdown(f"""
- Across **{insights['games_analyzed']:,} games** ({insights['date_range']}), teams averaged
  **{insights['avg_total_points'] / 2:.1f} points** per game; game totals have a standard deviation of
  **{insights['std_total_points']} points**, which is why totals betting is hard.
- **Home teams won {insights['home_win_pct']:.1%}** of games by an average of
  **{insights['avg_home_margin']:+.1f} points**. Our model's estimate, holding team strength fixed, is
  **{ratings.home_court:+.1f} points**.
- The box-score stat most correlated with winning is **{eda.STAT_LABELS.get(top_stat, top_stat)}**
  (r = {corr[top_stat]:.2f}). Shooting efficiency matters more than volume (see 🔗 Correlations).
- **{insights['blowout_outliers_count']} games** were statistical outliers by margin (IQR rule).
""")

    st.subheader("Data source")
    st.markdown(f"""
| | |
|---|---|
| **API** | ESPN public NBA scoreboard: `{SCOREBOARD_URL}` |
| **Format** | JSON: `events[] → competitions[0] → competitors[home, away] → statistics[]` |
| **Retrieval** | One request per month of history, one per upcoming day; cached in SQLite (`data/sportsbook.db`) |
| **Grain** | Raw: 1 row per game ({len(g):,} rows). Reshaped: 1 row per team per game ({len(raw_tg):,} rows) |
| **Excluded** | Preseason and exhibition games |
""")
    with st.expander("Preview raw data"):
        st.dataframe(g.drop(columns=["fetched_at"], errors="ignore").head(200), width="stretch")
        st.download_button("Download filtered games (CSV)", g.to_csv(index=False), "nba_games.csv", "text/csv")

# ---------------------------------------------------------------- data quality
with tabs[1]:
    st.subheader("Missing data")
    left, right = st.columns(2)
    with left:
        st.markdown("**Raw games table** (all statuses)")
        miss = eda.missing_report(g.drop(columns=["fetched_at"], errors="ignore"))
        st.dataframe(miss, width="stretch")
    with right:
        st.markdown("**Why values are missing, and how we handle them**")
        st.markdown(f"""
- **Scores missing** → the game hasn't been played yet. These are the games you can bet on;
  they are excluded from the analysis but drive the 🏀 Sportsbook.
- **All-Star Game** → ESPN labels the All-Star tournament (teams "STARS", "STRIPES", "WORLD") as
  regular season. Our missing-data check flagged these games because they had **no box scores**, so
  we now drop any "team" with fewer than 10 games. That leaves exactly the 30 NBA teams.
- **Attendance missing** → ESPN only reports it after games, and not always. It isn't used in modeling.
- **Box-score stats missing on a completed game** → the feed sometimes omits team stats.
  Treatment: **dropped {treatment['rows_dropped_no_boxscore']} team-game rows** with no box score from
  the stats analysis (their final scores still count for ratings), and
  **imputed {treatment['cells_imputed_team_median']} isolated cells** with that team's median.
""")

    st.subheader("Outliers")
    metric = st.radio("Variable", ["total_pts", "margin"], horizontal=True,
                      format_func={"total_pts": "Game total points", "margin": "Home margin"}.get)
    k = st.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.25,
                  help="A value is an outlier if it's below Q1 - k·IQR or above Q3 + k·IQR.")
    cg = eda.completed_games(done)
    out = eda.outlier_table(done, metric, k)
    o1, o2 = st.columns([3, 2])
    o1.plotly_chart(charts.boxplot(cg, metric, eda.STAT_LABELS[metric], by="season_type"), width="stretch")
    with o2:
        st.metric("Outlier games", f"{len(out)} of {len(cg)} ({len(out) / len(cg):.1%})")
        wins = eda.winsorize(cg[metric])
        st.markdown(f"""
**Treatment: winsorizing** (capping at the 1st/99th percentiles)

| | Raw | Winsorized |
|---|---|---|
| Mean | {cg[metric].mean():.2f} | {wins.mean():.2f} |
| Std dev | {cg[metric].std():.2f} | {wins.std():.2f} |
| Max | {cg[metric].max():.0f} | {wins.max():.0f} |

Blowouts and overtime shootouts are **real games**, so we don't delete them. The ratings model uses
ridge regularization and many games per team, so a single outlier has limited influence.
""")
    st.dataframe(out.head(25), hide_index=True, width="stretch")

# ---------------------------------------------------------------- distributions
with tabs[2]:
    st.subheader("Descriptive statistics (per team per game)")
    st.dataframe(eda.descriptive_stats(tg), width="stretch")
    options = {"pts": "Points", "margin": "Point margin", "fg_pct": "FG %", "fg3_pct": "3PT %",
               "fg3a": "3PT attempted", "reb": "Rebounds", "ast": "Assists", "fta": "FT attempted"}
    d1, d2 = st.columns([1, 3])
    col = d1.selectbox("Variable", list(options), format_func=options.get)
    split = d1.radio("Split by", ["none", "home/away", "result"])
    plot_df = tg.assign(**{"home/away": tg["is_home"].map({1: "Home", 0: "Away"}),
                           "result": tg["won"].map({1: "Win", 0: "Loss"})})
    d2.plotly_chart(charts.histogram(plot_df, col, options[col], None if split == "none" else split),
                    width="stretch")
    totals = eda.completed_games(done)
    st.plotly_chart(charts.histogram(totals, "total_pts", "Game total points"), width="stretch")

# ---------------------------------------------------------------- correlations
with tabs[3]:
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True,
                      help="Spearman is rank-based and less sensitive to outliers.")
    st.plotly_chart(charts.corr_heatmap(eda.correlations(tg, method)), width="stretch")
    st.markdown("""
**Reading this:** *FG %* has the strongest link to winning of the shooting stats, while raw *3PT attempted*
is near zero, so **efficiency beats volume**. *FG made* and *Points* are mechanically related
(points are built from makes), so they're expected to correlate.
""")
    s1, s2 = st.columns(2)
    xvar = s1.selectbox("X variable", list(options), index=2, format_func=options.get)
    s1.plotly_chart(charts.scatter_with_fit(tg, xvar, "margin", options[xvar], "Point margin",
                                            hover=["team", "opponent", "game_date"]), width="stretch")
    team_avg = eda.team_summary(tg).reset_index()
    s2.caption("Team averages over the selected games: point margin is a near-perfect proxy for win %.")
    s2.plotly_chart(charts.scatter_with_fit(team_avg, "margin", "win_pct", "Avg point margin", "Win %",
                                            hover=["team"]), width="stretch")

# ---------------------------------------------------------------- teams
with tabs[4]:
    t1, t2 = st.columns([2, 3])
    t1.plotly_chart(charts.ratings_bar(ratings.ratings), width="stretch")
    with t2:
        st.markdown(f"""
**How ratings are built:** a ridge regression of `margin = home-court + rating_home − rating_away`
over **{ratings.n_games:,} completed games**, weighting recent games more (45-day half-life) and
last season's games at 35%. Estimated home-court edge: **{ratings.home_court:+.2f} pts**; typical
error (σ): **{ratings.sigma:.1f} pts**.
""")
        st.dataframe(eda.team_summary(tg), width="stretch",
                     column_config={"win_pct": st.column_config.ProgressColumn("win %", min_value=0, max_value=1, format="%.3f")})

# ---------------------------------------------------------------- model evaluation
with tabs[5]:
    st.markdown("""
A **walk-forward back-test**: every week we refit ratings using only games *before* that week, then
predict that week's games. This is an honest test, because the model never sees the future.
""")
    bt = run_backtest(games[games["completed"]])
    if bt.empty:
        st.info("Not enough games yet to back-test.")
    else:
        bt = bt[bt["season"].isin(seasons)]
        acc = ((bt["p_home"] > 0.5) == bt["home_won"]).mean()
        brier = ((bt["p_home"] - bt["home_won"]) ** 2).mean()
        mae = (bt["pred_margin"] - bt["actual_margin"]).abs().mean()
        tot_mae = (bt["pred_total"] - bt["actual_total"]).abs().mean()
        m = st.columns(4)
        m[0].metric("Winner picked correctly", f"{acc:.1%}", help="Always picking the home team: "
                    f"{bt['home_won'].mean():.1%}")
        m[1].metric("Brier score", f"{brier:.3f}", help="Mean squared error of probabilities; 0.25 = coin flip.")
        m[2].metric("Margin MAE", f"{mae:.1f} pts")
        m[3].metric("Total MAE", f"{tot_mae:.1f} pts")
        e1, e2 = st.columns(2)
        e1.plotly_chart(charts.calibration_plot(bt), width="stretch")
        e2.plotly_chart(charts.scatter_with_fit(bt, "pred_margin", "actual_margin", "Predicted home margin",
                                                "Actual home margin", hover=["home_team", "away_team", "game_date"]),
                        width="stretch")
        st.session_state["backtest_summary"] = {
            "games_tested": len(bt), "accuracy": round(acc, 3), "brier": round(brier, 3),
            "margin_mae": round(mae, 2), "total_mae": round(tot_mae, 2),
            "always_pick_home_accuracy": round(bt["home_won"].mean(), 3),
        }
