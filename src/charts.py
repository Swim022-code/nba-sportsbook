"""Plotly chart builders with one consistent, colorblind-checked palette."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GRAY = "#e34948", "#898781"
SERIES = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", RED]
# Diverging scale for correlations: red (negative) - neutral gray - blue (positive)
DIVERGING = [[0.0, RED], [0.5, "#f0efec"], [1.0, BLUE]]


def _style(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title, margin=dict(l=10, r=10, t=50 if title else 10, b=10),
        hoverlabel=dict(font_size=13), legend=dict(orientation="h", y=-0.15),
        bargap=0.15,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5)
    return fig


def histogram(df: pd.DataFrame, col: str, label: str, color: str | None = None) -> go.Figure:
    fig = px.histogram(df, x=col, color=color, nbins=40, barmode="overlay", opacity=0.75,
                       color_discrete_sequence=SERIES, labels={col: label})
    mean = df[col].mean()
    fig.add_vline(x=mean, line_dash="dash", line_color=GRAY,
                  annotation_text=f"mean {mean:.1f}", annotation_position="top right")
    return _style(fig, f"Distribution of {label.lower()}")


def boxplot(df: pd.DataFrame, col: str, label: str, by: str | None = None) -> go.Figure:
    fig = px.box(df, x=by, y=col, points="outliers", color_discrete_sequence=[BLUE], labels={col: label})
    return _style(fig, f"{label}: box plot (dots = IQR outliers)")


def corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=DIVERGING, aspect="auto")
    fig.update_traces(textfont_size=11, hovertemplate="%{y} vs %{x}: r = %{z:.3f}<extra></extra>")
    return _style(fig, "Correlation matrix (team-game level)")


def scatter_with_fit(df: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str,
                     hover: list[str] | None = None) -> go.Figure:
    d = df[[x, y, *(hover or [])]].dropna()
    fig = px.scatter(d, x=x, y=y, hover_data=hover, opacity=0.45,
                     color_discrete_sequence=[BLUE], labels={x: xlabel, y: ylabel})
    if len(d) > 2:
        slope, intercept = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 50)
        r = d[x].corr(d[y])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                 line=dict(color=ORANGE, width=2), name=f"OLS fit (r = {r:.2f})"))
    return _style(fig, f"{ylabel} vs {xlabel}")


def ratings_bar(ratings: pd.Series) -> go.Figure:
    s = ratings.sort_values()
    colors = [BLUE if v >= 0 else RED for v in s]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:+.1f} pts<extra></extra>"))
    fig.update_layout(height=max(400, 18 * len(s)), xaxis_title="Net rating (pts vs. average team)")
    return _style(fig, "Model power ratings")


def calibration_plot(bt: pd.DataFrame, bins: int = 10) -> go.Figure:
    d = bt.copy()
    d["bin"] = pd.cut(d["p_home"], np.linspace(0, 1, bins + 1))
    cal = d.groupby("bin", observed=True).agg(pred=("p_home", "mean"), actual=("home_won", "mean"),
                                              n=("home_won", "size")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color=GRAY, dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=cal["pred"], y=cal["actual"], mode="lines+markers", name="Our model",
                             marker=dict(size=10, color=BLUE), line=dict(color=BLUE, width=2),
                             customdata=cal["n"],
                             hovertemplate=("predicted %{x:.0%}<br>actual %{y:.0%}<br>"
                                            "%{customdata} games<extra></extra>")))
    fig.update_layout(xaxis_title="Predicted home win probability", yaxis_title="Actual home win rate",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    return _style(fig, "Calibration: do 70% favorites win 70% of the time?")


def line(df: pd.DataFrame, x: str, y: str, title: str, labels: dict | None = None, pct: bool = False,
         color: str | None = None) -> go.Figure:
    """Line chart; with `color`, each group (e.g. season) is its own unconnected line."""
    fig = px.line(df, x=x, y=y, color=color, markers=True, labels=labels, color_discrete_sequence=SERIES)
    fig.update_traces(line_width=2, marker_size=8)
    if pct:
        fig.update_yaxes(tickformat=".0%")
    return _style(fig, title)
