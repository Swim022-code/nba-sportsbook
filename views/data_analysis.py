"""Exploratory Data Analysis of NBA game data, plus model evaluation."""

import streamlit as st

from src import charts, eda, llm
from src.data_fetcher import SCOREBOARD_URL
from src.ui import backtest_summary, get_backtest, get_games, get_ratings, season_label, sidebar, stream_llm

games = get_games()
sidebar()
ratings = get_ratings(games)


def ai_explain(section: str, data: dict, key: str) -> None:
    """A small 'Explain this' button that sends this section's numbers to Claude."""
    if st.button("🤖 Explain this with AI", key=key, disabled=not llm.is_configured(),
                 help=None if llm.is_configured() else "Needs an Anthropic API key (see README)"):
        with st.container(border=True):
            stream_llm(llm.explain_findings, section, data)


st.title("📊 Exploratory Data Analysis")
st.caption("Use the filters to slice the data; every table, chart and insight below updates.")

# ---------------------------------------------------------------- filters
seasons_all = sorted(games["season"].unique())
f1, f2, f3 = st.columns(3)
seasons = f1.multiselect("Season", seasons_all, default=seasons_all, format_func=season_label)
game_types = {"regular-season": "Regular season", "play-in-season": "Play-in", "post-season": "Playoffs"}
types = f2.multiselect("Game type", list(game_types), default=list(game_types), format_func=game_types.get)
all_teams = sorted(set(games["home_team"]) | set(games["away_team"]))
teams = f3.multiselect("Teams (empty = all)", all_teams)

g = games[games["season"].isin(seasons) & games["season_type"].isin(types)]
if teams:
    g = g[g["home_team"].isin(teams) | g["away_team"].isin(teams)]
done = g[g["completed"]]
if len(done) < 20:
    st.warning("Fewer than 20 completed games match these filters. Widen your selection.")
    st.stop()

raw_tg = eda.to_team_games(done)
if teams:
    raw_tg = raw_tg[raw_tg["team"].isin(teams)]
tg, treatment = eda.treat_missing(raw_tg)
insights = eda.key_insights(done, tg)
win_corr = tg[["won", "fg_pct", "fg3_pct", "fg3m", "fg3a", "reb", "ast", "fta", "ftm", "ft_pct"]].corr()["won"]

tabs = st.tabs(["📋 Overview", "🧹 Data quality", "📈 Distributions & trends", "🔗 Correlations",
                "🏀 Teams", "🎯 Model evaluation"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Completed games", f"{insights['games_analyzed']:,}")
    c[1].metric("Avg game total", f"{insights['avg_total_points']} pts")
    c[2].metric("Home win %", f"{insights['home_win_pct']:.1%}")
    c[3].metric("Avg winning margin", f"{insights['avg_abs_margin']} pts")

    st.subheader("Key insights")
    hc = eda.home_court_stats(done)
    lo, hi = hc["margin_95ci"]
    st.markdown(f"""
- Across **{insights['games_analyzed']:,} games** ({insights['date_range']}), teams averaged
  **{insights['avg_total_points'] / 2:.1f} points**. Game totals vary a lot (standard deviation
  **{insights['std_total_points']} pts**), which makes totals hard to predict.
- **Home teams won {insights['home_win_pct']:.1%}** of games, by **{insights['avg_home_margin']:+.1f} pts** on average
  (95% CI {lo:+.1f} to {hi:+.1f}). After controlling for team strength, the model estimates **{ratings.home_court:+.1f} pts**.
- {eda.describe_correlations(win_corr)} **Shooting efficiency matters more than shot volume.**
- **{insights['blowout_outliers_count']} games** were statistical outliers by margin (IQR rule).
""")

    st.subheader("Data source & structure")
    st.markdown(f"""
| | |
|---|---|
| **API** | ESPN public NBA scoreboard: `{SCOREBOARD_URL}` (free, no key) |
| **Format** | JSON: `events[] → competitions[0] → competitors[home, away] → statistics[]` |
| **Retrieval** | One request per month of history, one per upcoming day. Cached in SQLite (`data/sportsbook.db`); finished months are never re-downloaded |
| **Grain** | Raw: 1 row per game ({len(g):,} rows, 43 columns). Reshaped: 1 row per team per game ({len(raw_tg):,} rows) |
| **Excluded** | Preseason games and the All-Star Game (see 🧹 Data quality) |
""")
    with st.expander("Preview the raw data"):
        st.dataframe(g.drop(columns=["fetched_at"], errors="ignore").head(200), width="stretch")
        st.download_button("⬇️ Download filtered games (CSV)", g.to_csv(index=False), "nba_games.csv", "text/csv")

# ---------------------------------------------------------------- data quality
with tabs[1]:
    st.subheader("Missing data")
    left, right = st.columns(2)
    with left:
        which = st.radio("Table", ["Completed games", "Upcoming / postponed games"], horizontal=True)
        subset = done if which == "Completed games" else g[~g["completed"]]
        miss = eda.missing_report(subset.drop(columns=["fetched_at"], errors="ignore"))
        if miss.empty:
            st.success(f"No missing values in {len(subset):,} rows.")
        else:
            st.dataframe(miss, width="stretch")
        st.caption(f"{len(subset):,} rows. Status counts: "
                   + ", ".join(f"{k.replace('STATUS_', '').title()} {v}" for k, v in subset["status"].value_counts().items()))
    with right:
        st.markdown("**Why values are missing, and how we treat them**")
        st.markdown(f"""
- **Scores missing** → the game hasn't been played. These rows are excluded from the analysis but
  feed the 🏀 Sportsbook. **Postponed** games never get a score, so bets on them are **voided and refunded**.
- **All-Star Game** → ESPN labels the All-Star tournament (teams "STARS", "STRIPES", "WORLD") as
  regular season. Our missing-data check flagged these games because they had **no box scores**, so we
  drop any "team" with fewer than 10 games. Exactly **30 NBA teams** remain.
- **Attendance missing** → reported only after games, and not always. It isn't used in modeling, so it's left as-is.
- **Box-score gaps on completed games** → safeguard: drop rows with no box score at all
  (**{treatment['rows_dropped_no_boxscore']} dropped**) and fill isolated gaps with the team's median
  (**{treatment['cells_imputed_team_median']} imputed**).
""")

    st.subheader("Outliers")
    o_ctrl, o_info = st.columns([1, 1])
    metric = o_ctrl.radio("Variable", ["total_pts", "margin"], horizontal=True,
                          format_func={"total_pts": "Game total points", "margin": "Home margin"}.get)
    k = o_ctrl.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.25,
                      help="A value is an outlier if it's below Q1 − k·IQR or above Q3 + k·IQR.")
    cg = eda.completed_games(done)
    out = eda.outlier_table(done, metric, k)
    wins = eda.winsorize(cg[metric])
    o_info.metric("Outlier games", f"{len(out)} of {len(cg):,} ({len(out) / len(cg):.1%})")
    p1, p2 = st.columns([3, 2])
    p1.plotly_chart(charts.boxplot(cg.assign(season_type=cg["season_type"].map(game_types)),
                                   metric, eda.STAT_LABELS[metric], by="season_type"), width="stretch")
    with p2:
        st.markdown(f"""
**Treatment: winsorizing** (capping at the 1st/99th percentiles)

| | Raw | Winsorized |
|---|---|---|
| Mean | {cg[metric].mean():.2f} | {wins.mean():.2f} |
| Std dev | {cg[metric].std():.2f} | {wins.std():.2f} |
| Min / Max | {cg[metric].min():.0f} / {cg[metric].max():.0f} | {wins.min():.0f} / {wins.max():.0f} |

Blowouts and overtime shootouts are **real games**, so we keep them. Winsorizing barely moves the
mean, which confirms they don't distort averages, and the ratings model's ridge penalty limits any
single game's influence.
""")
    st.dataframe(out.head(25), hide_index=True, width="stretch")
    ai_explain("Outliers", {"variable": metric, "k": k, "n_outliers": len(out), "n_games": len(cg),
                            "raw_mean": cg[metric].mean(), "winsorized_mean": wins.mean(),
                            "raw_std": cg[metric].std(), "winsorized_std": wins.std(),
                            "top_outliers": out.head(5).astype(str).to_dict("records")}, "ai_outliers")

# ---------------------------------------------------------------- distributions
with tabs[2]:
    st.subheader("Descriptive statistics (per team per game)")
    st.dataframe(eda.descriptive_stats(tg), width="stretch")
    options = {"pts": "Points", "margin": "Point margin", "fg_pct": "FG %", "fg3_pct": "3PT %",
               "fg3a": "3PT attempted", "reb": "Rebounds", "ast": "Assists", "fta": "FT attempted"}
    d1, d2 = st.columns([1, 3])
    col = d1.selectbox("Variable", list(options), format_func=options.get)
    split = d1.radio("Compare", ["none", "home/away", "result"],
                     format_func={"none": "No split", "home/away": "Home vs away", "result": "Wins vs losses"}.get)
    plot_df = tg.assign(**{"home/away": tg["is_home"].map({1: "Home", 0: "Away"}),
                           "result": tg["won"].map({1: "Win", 0: "Loss"})})
    d2.plotly_chart(charts.histogram(plot_df, col, options[col], None if split == "none" else split),
                    width="stretch")
    if split != "none":
        grp = plot_df.groupby(split)[col].mean()
        d1.markdown("**Group means**  \n" + "  \n".join(f"{i}: **{v:.2f}**" for i, v in grp.items()))

    st.subheader("Trends over time")
    trend = eda.scoring_trend(done)
    t1, t2 = st.columns(2)
    t1.plotly_chart(charts.line(trend, "month", "avg_total", "Average game total by month",
                                {"avg_total": "Avg total points", "month": ""}), width="stretch")
    t2.plotly_chart(charts.line(trend, "month", "home_win_pct", "Home win % by month",
                                {"home_win_pct": "Home win %", "month": ""}, pct=True), width="stretch")
    playoff = eda.completed_games(done).groupby("season_type")["total_pts"].mean()
    if {"regular-season", "post-season"} <= set(playoff.index):
        st.info(f"Playoff games average **{playoff['post-season']:.1f}** total points vs "
                f"**{playoff['regular-season']:.1f}** in the regular season "
                f"({playoff['post-season'] - playoff['regular-season']:+.1f}): tighter defense in the playoffs.", icon="💡")

# ---------------------------------------------------------------- correlations
with tabs[3]:
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True,
                      help="Spearman uses ranks, so it is less sensitive to outliers.")
    corr = eda.correlations(tg, method)
    st.plotly_chart(charts.corr_heatmap(corr), width="stretch")
    st.markdown(f"""
**Reading this:** {eda.describe_correlations(win_corr)} 3PT % (r = {win_corr['fg3_pct']:.2f}) matters far more
than 3PT attempts (r = {win_corr['fg3a']:.2f}), so **efficiency beats volume**. Some pairs are related by
definition (FG made → Points, FT made ↔ FT attempted), so their high correlations are expected, not insights.
*Correlation is not causation:* winning teams also get more free throws late in games, for example.
""")
    s1, s2 = st.columns(2)
    xvar = s1.selectbox("X variable", list(options), index=2, format_func=options.get)
    s1.plotly_chart(charts.scatter_with_fit(tg, xvar, "margin", options[xvar], "Point margin",
                                            hover=["team", "opponent", "game_date"]), width="stretch")
    team_avg = eda.team_summary(tg).reset_index()
    s2.caption("Team averages over the selected games: point margin is a near-perfect proxy for win %.")
    s2.plotly_chart(charts.scatter_with_fit(team_avg, "margin", "win_pct", "Avg point margin", "Win %",
                                            hover=["team"]), width="stretch")
    ai_explain("Correlations with winning", {"method": method, "correlation_with_won": win_corr.round(3).to_dict(),
                                             "n_team_games": len(tg)}, "ai_corr")

# ---------------------------------------------------------------- teams
with tabs[4]:
    t1, t2 = st.columns([2, 3])
    t1.plotly_chart(charts.ratings_bar(ratings.ratings), width="stretch")
    with t2:
        st.markdown(f"""
**How ratings are built:** a weighted ridge regression of
`margin = home-court + rating_home − rating_away` over **{ratings.n_games:,} completed games**.
Recent games get more weight (45-day half-life) and last season's games count at 35%.
Estimated home-court edge **{ratings.home_court:+.2f} pts**; typical prediction error (σ) **{ratings.sigma:.1f} pts**.
""")
        st.dataframe(eda.team_summary(tg), width="stretch",
                     column_config={"win_pct": st.column_config.ProgressColumn("win %", min_value=0, max_value=1,
                                                                               format="%.3f")})

# ---------------------------------------------------------------- model evaluation
with tabs[5]:
    st.markdown("""
A **walk-forward back-test**: each week the ratings are refit using only games played *before* that week,
then used to predict that week's games. The model never sees the future, so this is an honest test.
""")
    bt = get_backtest(games)
    bt = bt[bt["season"].isin(seasons)] if not bt.empty else bt
    if bt.empty:
        st.info("Not enough games to back-test with these filters.")
    else:
        summary = backtest_summary(bt)
        acc, brier, baseline = summary["accuracy"], summary["brier"], summary["always_pick_home_accuracy"]
        mae, tot_mae = summary["margin_mae"], summary["total_mae"]
        m = st.columns(4)
        m[0].metric("Winner picked correctly", f"{acc:.1%}", f"{acc - baseline:+.1%} vs always-home",
                    help=f"Baseline (always pick the home team): {baseline:.1%}")
        m[1].metric("Brier score", f"{brier:.3f}", f"{brier - 0.25:+.3f} vs coin flip", delta_color="inverse",
                    help="Mean squared error of the win probabilities. Lower is better; 0.25 = coin flip.")
        m[2].metric("Margin error (MAE)", f"{mae:.1f} pts")
        m[3].metric("Total error (MAE)", f"{tot_mae:.1f} pts")
        e1, e2 = st.columns(2)
        e1.plotly_chart(charts.calibration_plot(bt), width="stretch")
        e2.plotly_chart(charts.scatter_with_fit(bt, "pred_margin", "actual_margin", "Predicted home margin",
                                                "Actual home margin", hover=["home_team", "away_team", "game_date"]),
                        width="stretch")
        ai_explain("Model evaluation (walk-forward back-test)", summary, "ai_model")
