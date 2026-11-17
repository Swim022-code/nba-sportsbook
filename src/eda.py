"""Exploratory data analysis helpers (pure pandas - no Streamlit here, so it's testable).

Main idea: reshape games (one row per game) into team-games (one row per team per
game) so we can ask "what box-score stats go with winning?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BOX_STATS = ["pts", "reb", "ast", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct"]
STAT_LABELS = {
    "pts": "Points", "reb": "Rebounds", "ast": "Assists", "fgm": "FG made", "fga": "FG attempted",
    "fg_pct": "FG %", "fg3m": "3PT made", "fg3a": "3PT attempted", "fg3_pct": "3PT %",
    "ftm": "FT made", "fta": "FT attempted", "ft_pct": "FT %", "margin": "Point margin",
    "won": "Won (1/0)", "opp_pts": "Points allowed", "total_pts": "Game total",
}


def completed_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games[games["completed"]].copy()
    g["margin"] = g["home_score"] - g["away_score"]
    g["total_pts"] = g["home_score"] + g["away_score"]
    g["home_won"] = (g["margin"] > 0).astype(int)
    return g


def to_team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per completed game, with the opponent's points attached."""
    g = completed_games(games)
    frames = []
    for side, opp in (("home", "away"), ("away", "home")):
        cols = {f"{side}_{s}": s for s in BOX_STATS}
        part = g[["game_id", "game_date", "season", "season_type", *cols]].rename(columns=cols)
        part["team"] = g[f"{side}_team"]
        part["opponent"] = g[f"{opp}_team"]
        part["is_home"] = int(side == "home")
        part["opp_pts"] = g[f"{opp}_score"]
        part["pts"] = g[f"{side}_score"]          # final score is authoritative
        frames.append(part)
    tg = pd.concat(frames, ignore_index=True)
    tg["margin"] = tg["pts"] - tg["opp_pts"]
    tg["won"] = (tg["margin"] > 0).astype(int)
    return tg


# ---------------------------------------------------------------- data quality
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and % of missing values per column (only columns with any missing)."""
    miss = df.isna().sum()
    rep = pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(2)})
    return rep[rep["missing"] > 0].sort_values("missing", ascending=False)


def treat_missing(team_games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Treatment: games with NO box score are dropped from stat analysis (scores stay);
    any remaining isolated gaps are filled with that team's median."""
    before = len(team_games)
    tg = team_games.dropna(subset=["fga", "reb"], how="all").copy()
    dropped = before - len(tg)
    filled = int(tg[BOX_STATS].isna().sum().sum())
    for col in BOX_STATS:
        tg[col] = tg[col].fillna(tg.groupby("team")[col].transform("median"))
    return tg, {"rows_dropped_no_boxscore": dropped, "cells_imputed_team_median": filled}


def iqr_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Boolean mask: True where a value is outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


def outlier_table(games: pd.DataFrame, column: str = "total_pts", k: float = 1.5) -> pd.DataFrame:
    g = completed_games(games)
    mask = iqr_outliers(g[column], k)
    out = g.loc[mask, ["game_date", "away_team", "away_score", "home_team", "home_score", "margin", "total_pts"]]
    return out.sort_values(column, key=np.abs, ascending=False)


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Cap extreme values at the given percentiles (keeps rows, limits their influence)."""
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi)


# ---------------------------------------------------------------- summaries
def descriptive_stats(team_games: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or ["pts", "opp_pts", "margin", "reb", "ast", "fg_pct", "fg3_pct", "fg3a", "ft_pct", "fta"]
    desc = team_games[cols].describe().T
    desc["skew"] = team_games[cols].skew()
    desc.index = [STAT_LABELS.get(c, c) for c in desc.index]
    return desc.round(2)


def correlations(team_games: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    cols = ["won", "margin", "pts", "reb", "ast", "fg_pct", "fg3m", "fg3a", "fg3_pct", "ftm", "fta", "ft_pct"]
    corr = team_games[cols].corr(method=method)
    labels = [STAT_LABELS.get(c, c) for c in cols]
    corr.index, corr.columns = labels, labels
    return corr.round(3)


def team_summary(team_games: pd.DataFrame) -> pd.DataFrame:
    t = team_games.groupby("team").agg(
        games=("won", "size"), wins=("won", "sum"), ppg=("pts", "mean"), opp_ppg=("opp_pts", "mean"),
        margin=("margin", "mean"), fg_pct=("fg_pct", "mean"), fg3_pct=("fg3_pct", "mean"),
        fg3a=("fg3a", "mean"), reb=("reb", "mean"), ast=("ast", "mean"),
    )
    t["win_pct"] = t["wins"] / t["games"]
    return t.sort_values("margin", ascending=False).round(3)


def home_court_stats(games: pd.DataFrame) -> dict:
    g = completed_games(games)
    g = g[~g["neutral_site"]]
    margin = g["margin"]
    se = margin.std() / np.sqrt(len(margin)) if len(margin) > 1 else float("nan")
    return {
        "games": int(len(g)),
        "home_win_pct": float(g["home_won"].mean()),
        "avg_home_margin": float(margin.mean()),
        "margin_95ci": (float(margin.mean() - 1.96 * se), float(margin.mean() + 1.96 * se)),
    }


def key_insights(games: pd.DataFrame, team_games: pd.DataFrame) -> dict:
    """Compact, JSON-friendly facts - also what we send to the LLM as context."""
    g = completed_games(games)
    corr = team_games[["won", "fg_pct", "fg3_pct", "reb", "ast", "fta", "fg3a"]].corr()["won"].drop("won")
    teams = team_summary(team_games)
    hc = home_court_stats(games)
    outliers = outlier_table(games, "margin")
    return {
        "games_analyzed": int(len(g)),
        "date_range": f"{g['game_date'].min():%Y-%m-%d} to {g['game_date'].max():%Y-%m-%d}",
        "avg_total_points": round(float(g["total_pts"].mean()), 1),
        "std_total_points": round(float(g["total_pts"].std()), 1),
        "avg_abs_margin": round(float(g["margin"].abs().mean()), 1),
        "home_win_pct": round(hc["home_win_pct"], 3),
        "avg_home_margin": round(hc["avg_home_margin"], 2),
        "correlation_with_winning": corr.round(3).sort_values(ascending=False).to_dict(),
        "best_teams_by_margin": teams.head(5)[["win_pct", "margin"]].round(3).to_dict("index"),
        "worst_teams_by_margin": teams.tail(5)[["win_pct", "margin"]].round(3).to_dict("index"),
        "blowout_outliers_count": int(len(outliers)),
        "biggest_blowout": (outliers.iloc[0][["game_date", "away_team", "away_score", "home_team", "home_score"]]
                            .astype(str).to_dict() if len(outliers) else None),
    }


def scoring_trend(games: pd.DataFrame, min_games: int = 1) -> pd.DataFrame:
    """Average game total and home win % by season and calendar month.

    Months with fewer than `min_games` games (e.g. a 5-game NBA Finals month) are
    dropped because their averages are too noisy to read as a trend.
    """
    g = completed_games(games)
    g["month"] = pd.to_datetime(g["game_date"]).dt.to_period("M").dt.to_timestamp()
    by = ["season", "month"] if "season" in g else ["month"]
    out = (g.groupby(by)
             .agg(games=("game_id", "size"), avg_total=("total_pts", "mean"),
                  home_win_pct=("home_won", "mean"))
             .reset_index())
    return out[out["games"] >= min_games].reset_index(drop=True)


def describe_correlations(corr_with_win: pd.Series) -> str:
    """One plain-English sentence summarising which stats go with winning."""
    s = corr_with_win.drop(labels=["won", "margin", "pts"], errors="ignore").sort_values(ascending=False)
    strongest, weakest = s.index[0], s.abs().idxmin()
    return (f"**{STAT_LABELS.get(strongest, strongest)}** has the strongest link to winning "
            f"(r = {s.iloc[0]:.2f}), while **{STAT_LABELS.get(weakest, weakest)}** has almost none "
            f"(r = {s[weakest]:.2f}).")
