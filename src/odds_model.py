"""Our own odds-making model ("the house").

ESPN's free API doesn't include betting lines, so the sportsbook sets its own,
which is the data-science core of the project:

1. Team power ratings: fit   margin = HCA + r_home - r_away   by ridge
   least squares over completed games (r = points better than an average team,
   HCA = home-court advantage, both estimated from data). Recent games count
   more (exponential decay) and last season's games are down-weighted.
2. Win probability:  P(home wins) = Phi(predicted_margin / sigma), where sigma
   is the standard deviation of the model's residuals.
3. Lines:
   * spread    = predicted margin rounded to the nearest 0.5, priced -110/-110
   * total     = each team's expected points from offense/defense ratings, -110/-110
   * moneyline = fair probabilities with a ~4.5% house margin ("vig") added,
     converted to American odds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

VIG = 0.045                    # bookmaker overround on the moneyline
STANDARD_JUICE = -110          # price on spreads and totals
RIDGE_LAMBDA = 2.0             # shrinks ratings toward 0 when a team has few games
HALF_LIFE_DAYS = 45            # recent games matter more
PRIOR_SEASON_WEIGHT = 0.35     # last season's games count ~1/3 as much


# ---------------------------------------------------------------- odds math
def american_to_decimal(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def american_to_implied_prob(odds: int) -> float:
    return 1 / american_to_decimal(odds)


def prob_to_american(p: float) -> int:
    """Convert a probability to American odds, rounded to the nearest 5."""
    p = min(max(p, 0.01), 0.99)
    raw = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    rounded = int(5 * round(raw / 5))
    if -100 < rounded < 100:  # -100..+100 are not quoted; snap to even money
        rounded = -100 if p >= 0.5 else 100
    return rounded


def payout_for(stake: float, odds: int) -> float:
    """Total returned on a winning bet (stake + profit)."""
    return round(stake * american_to_decimal(odds), 2)


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def round_half(x: float) -> float:
    return round(x * 2) / 2


# ---------------------------------------------------------------- ratings
@dataclass
class TeamRatings:
    ratings: pd.Series          # net rating (points vs. average team), indexed by team
    offense: pd.Series          # expected points scored vs. average defense
    defense: pd.Series          # expected points allowed vs. average offense
    home_court: float           # points
    sigma: float                # std dev of margin residuals
    league_avg_pts: float
    n_games: int


def _weights(games: pd.DataFrame, as_of: pd.Timestamp, current_season: int) -> np.ndarray:
    age_days = (as_of - games["game_date"]).dt.days.clip(lower=0).to_numpy()
    w = 0.5 ** (age_days / HALF_LIFE_DAYS)
    w = np.where(games["season"].to_numpy() == current_season, w, w * PRIOR_SEASON_WEIGHT)
    return np.maximum(w, 1e-3)


def _ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, n_free: int) -> np.ndarray:
    """Weighted ridge regression; the last column(s) beyond n_free are unpenalized."""
    W = np.sqrt(w)[:, None]
    Xw, yw = X * W, y * W[:, 0]
    penalty = np.eye(X.shape[1]) * RIDGE_LAMBDA
    penalty[n_free:, n_free:] = 0
    return np.linalg.solve(Xw.T @ Xw + penalty, Xw.T @ yw)


def fit_ratings(completed: pd.DataFrame, as_of: pd.Timestamp | None = None) -> TeamRatings:
    """Fit team ratings from completed games (needs home/away team + score)."""
    g = completed.dropna(subset=["home_score", "away_score"]).copy()
    if len(g) < 10:
        raise ValueError("Need at least 10 completed games to fit ratings")
    g["game_date"] = pd.to_datetime(g["game_date"])
    as_of = as_of or g["game_date"].max()
    current_season = int(g.loc[g["game_date"].idxmax(), "season"])

    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(g), len(teams)
    hi = g["home_team"].map(idx).to_numpy()
    ai = g["away_team"].map(idx).to_numpy()
    rows = np.arange(n)
    home_flag = (~g["neutral_site"].astype(bool)).astype(float).to_numpy()
    w = _weights(g, as_of, current_season)

    # Margin model: margin = r_home - r_away + HCA * home_flag
    X = np.zeros((n, k + 1))
    X[rows, hi] += 1
    X[rows, ai] -= 1
    X[:, k] = home_flag
    margin = (g["home_score"] - g["away_score"]).to_numpy()
    beta = _ridge(X, margin, w, n_free=k)
    ratings = pd.Series(beta[:k], index=teams)
    hca = float(beta[k])
    resid = margin - X @ beta
    sigma = float(np.sqrt(np.average(resid**2, weights=w)))

    # Points model: pts_scored = league_avg + off_team - def_opp (+/- HCA/2)
    league_avg = float(np.average(np.r_[g["home_score"], g["away_score"]], weights=np.r_[w, w]))
    Xp = np.zeros((2 * n, 2 * k + 1))
    # home scoring rows
    Xp[rows, hi] = 1
    Xp[rows, k + ai] = 1
    Xp[rows, 2 * k] = home_flag / 2
    # away scoring rows
    Xp[n + rows, ai] = 1
    Xp[n + rows, k + hi] = 1
    Xp[n + rows, 2 * k] = -home_flag / 2
    yp = np.r_[g["home_score"], g["away_score"]] - league_avg
    bp = _ridge(Xp, yp, np.r_[w, w], n_free=2 * k)
    offense = pd.Series(league_avg + bp[:k], index=teams)
    defense = pd.Series(league_avg + bp[k:2 * k], index=teams)  # points allowed

    return TeamRatings(ratings.sort_values(ascending=False), offense, defense,
                       hca, sigma, league_avg, n)


# ---------------------------------------------------------------- lines
def predict_game(r: TeamRatings, home: str, away: str, neutral: bool = False) -> dict:
    """Predicted margin, total and win probability for one matchup."""
    hca = 0.0 if neutral else r.home_court
    margin = r.ratings.get(home, 0.0) - r.ratings.get(away, 0.0) + hca
    home_pts = (r.offense.get(home, r.league_avg_pts) + r.defense.get(away, r.league_avg_pts)
                - r.league_avg_pts + hca / 2)
    away_pts = (r.offense.get(away, r.league_avg_pts) + r.defense.get(home, r.league_avg_pts)
                - r.league_avg_pts - hca / 2)
    p_home = normal_cdf(margin / r.sigma)
    return {"pred_margin": margin, "pred_total": home_pts + away_pts,
            "pred_home_pts": home_pts, "pred_away_pts": away_pts, "p_home": p_home}


def make_lines(r: TeamRatings, home: str, away: str, neutral: bool = False) -> dict:
    """The full betting menu for one game."""
    pred = predict_game(r, home, away, neutral)
    p_home = pred["p_home"]
    # Spread is quoted from the home team's view: -5.5 means home must win by 6+.
    home_spread = -round_half(pred["pred_margin"])
    if home_spread == 0:
        home_spread = -0.5 if p_home >= 0.5 else 0.5   # no pick'em, avoids pushes
    return {
        **pred,
        "home_ml": prob_to_american(p_home * (1 + VIG)),
        "away_ml": prob_to_american((1 - p_home) * (1 + VIG)),
        "home_spread": home_spread,
        "away_spread": -home_spread,
        "spread_odds": STANDARD_JUICE,
        "total": round_half(pred["pred_total"]),
        "total_odds": STANDARD_JUICE,
    }


def backtest(completed: pd.DataFrame, min_train_games: int = 150, step_days: int = 7) -> pd.DataFrame:
    """Walk-forward test: re-fit weekly using only earlier games, predict the next week.

    This shows how good our lines are WITHOUT peeking at the future.
    """
    g = completed.dropna(subset=["home_score", "away_score"]).copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g.sort_values("game_date")
    out = []
    dates = g["game_date"].sort_values().unique()
    if len(g) <= min_train_games:
        return pd.DataFrame()
    cutoff = pd.Timestamp(dates[0]) + pd.Timedelta(days=1)
    while len(g[g["game_date"] < cutoff]) < min_train_games:
        cutoff += pd.Timedelta(days=1)
    while cutoff <= g["game_date"].max():
        train = g[g["game_date"] < cutoff]
        test = g[(g["game_date"] >= cutoff) & (g["game_date"] < cutoff + pd.Timedelta(days=step_days))]
        if not test.empty:
            r = fit_ratings(train, as_of=cutoff)
            for _, row in test.iterrows():
                p = predict_game(r, row.home_team, row.away_team, bool(row.neutral_site))
                actual_margin = row.home_score - row.away_score
                out.append({
                    "game_id": row.game_id, "game_date": row.game_date, "season": row.season,
                    "home_team": row.home_team, "away_team": row.away_team,
                    "pred_margin": p["pred_margin"], "actual_margin": actual_margin,
                    "pred_total": p["pred_total"], "actual_total": row.home_score + row.away_score,
                    "p_home": p["p_home"], "home_won": actual_margin > 0,
                })
        cutoff += pd.Timedelta(days=step_days)
    return pd.DataFrame(out)
