"""Futures: championship odds from a Monte Carlo simulation of the rest of the season.

Method (N = 5,000 simulated seasons, vectorised with NumPy):
1. Regular season: each team adds Binomial(games left, p) wins to its current record,
   where p is its chance to beat an average team (half home, half away games),
   from the power ratings.
2. Seeding: rank teams by wins within each conference (random tie-breaks).
   Seeds 1-6 qualify; 7-10 play the play-in (7v8 winner = 7th seed; loser plays the
   9v10 winner for the 8th seed).
3. Playoffs: best-of-7 series (higher seed has home court in games 1, 2, 5, 7), each
   game decided by the ratings model with home-court advantage.
The share of simulations each team wins = its title probability. Odds then add a
futures margin when the house edge is on (sportsbooks hold much more on futures).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.odds_model import TeamRatings

EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NY", "ORL", "PHI", "TOR", "WSH"}
WEST = {"DAL", "DEN", "GS", "HOU", "LAC", "LAL", "MEM", "MIN", "NO", "OKC", "PHX", "POR", "SA", "SAC", "UTAH"}
SEASON_GAMES = 82
FUTURES_OVERROUND = 1.25   # house edge ON: implied title probabilities sum to ~125%
LONGSHOT_FLOOR = 0.001     # longest odds offered: about +100000
SERIES_HOME = np.array([1, 1, 0, 0, 1, 0, 1], dtype=bool)  # higher seed's home games


def standings(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Current regular-season W-L for each team."""
    g = games[(games["season"] == season) & (games["season_type"] == "regular-season") & games["completed"]]
    home = pd.DataFrame({"team": g["home_team"], "win": g["home_score"] > g["away_score"]})
    away = pd.DataFrame({"team": g["away_team"], "win": g["away_score"] > g["home_score"]})
    s = pd.concat([home, away]).groupby("team")["win"].agg(wins="sum", gp="size")
    s["losses"] = s["gp"] - s["wins"]
    s["conference"] = ["East" if t in EAST else "West" for t in s.index]
    return s.reset_index()


def american_long(p: float) -> int:
    """American odds for longshots, rounded the way books quote futures."""
    raw = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    step = 5 if abs(raw) < 500 else 25 if abs(raw) < 2000 else 100 if abs(raw) < 10000 else 500
    return int(step * round(raw / step))


def simulate(r: TeamRatings, table: pd.DataFrame, n_sims: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = sorted(set(table["team"]) & (EAST | WEST))
    idx = {t: i for i, t in enumerate(teams)}
    t = table.set_index("team").reindex(teams)
    rating = r.ratings.reindex(teams).fillna(0).to_numpy()
    hca, sigma = r.home_court, r.sigma

    # 1. rest of regular season
    p_game = 0.5 * (ndtr((rating + hca) / sigma) + ndtr((rating - hca) / sigma))
    left = np.clip(SEASON_GAMES - t["gp"].fillna(0).to_numpy(), 0, None).astype(int)
    wins = t["wins"].fillna(0).to_numpy() + rng.binomial(left, p_game, size=(n_sims, len(teams)))
    score = wins + rng.random(wins.shape) * 0.5  # random tie-break

    def game(home, away):
        """Vectorised single game: True where `home` wins."""
        return rng.random(len(home)) < ndtr((rating[home] - rating[away] + hca) / sigma)

    def series(hi, lo):
        """Best-of-7; `hi` has home court. Majority of 7 games decides the winner."""
        d = rating[hi] - rating[lo]
        p = np.where(SERIES_HOME[:, None], ndtr((d + hca) / sigma), ndtr((d - hca) / sigma))
        hi_wins = (rng.random((7, len(hi))) < p).sum(axis=0)
        return np.where(hi_wins >= 4, hi, lo)

    rows = np.arange(n_sims)
    champs, finalists, playoffs = [], [], np.zeros((n_sims, len(teams)), dtype=bool)
    conf_champ = {}
    for conf, members in (("East", EAST), ("West", WEST)):
        cidx = np.array([idx[x] for x in teams if x in members])
        order = cidx[np.argsort(-score[:, cidx], axis=1)]  # (n_sims, 15) best first
        seed_rank = np.empty((n_sims, len(teams)), dtype=int)
        seed_rank[rows[:, None], order] = np.arange(len(cidx))
        s = [order[:, k] for k in range(10)]  # s[0] = 1st seed
        w78 = game(s[6], s[7])
        seventh, loser78 = np.where(w78, s[6], s[7]), np.where(w78, s[7], s[6])
        ninth_tenth = np.where(game(s[8], s[9]), s[8], s[9])
        eighth = np.where(game(loser78, ninth_tenth), loser78, ninth_tenth)
        field = [s[0], s[1], s[2], s[3], s[4], s[5], seventh, eighth]
        for tm in field:
            playoffs[rows, tm] = True

        def by_seed(a, b, rank=seed_rank):
            better = rank[rows, a] <= rank[rows, b]
            return np.where(better, a, b), np.where(better, b, a)

        r1 = [series(*by_seed(field[0], field[7])), series(*by_seed(field[3], field[4])),
              series(*by_seed(field[2], field[5])), series(*by_seed(field[1], field[6]))]
        semis = [series(*by_seed(r1[0], r1[1])), series(*by_seed(r1[3], r1[2]))]
        conf_champ[conf] = series(*by_seed(semis[0], semis[1]))
        finalists.append(conf_champ[conf])

    e, w = conf_champ["East"], conf_champ["West"]
    e_better = score[rows, e] >= score[rows, w]
    champs = series(np.where(e_better, e, w), np.where(e_better, w, e))

    out = pd.DataFrame({
        "team": teams,
        "conference": ["East" if x in EAST else "West" for x in teams],
        "wins": t["wins"].fillna(0).astype(int).to_numpy(), "losses": t["losses"].fillna(0).astype(int).to_numpy(),
        "rating": rating, "proj_wins": wins.mean(axis=0),
        "p_playoffs": playoffs.mean(axis=0),
        "p_conf": np.bincount(np.concatenate(finalists), minlength=len(teams)) / n_sims,
        "p_title": np.bincount(champs, minlength=len(teams)) / n_sims,
    })
    return out.sort_values("p_title", ascending=False).reset_index(drop=True)


def futures_odds(sim: pd.DataFrame, house_edge: bool) -> pd.DataFrame:
    """Add American odds for the championship market. Every team is offered; teams that
    never won in the simulation get the longest price (about +100000)."""
    df = sim.copy()
    factor = FUTURES_OVERROUND if house_edge else 1.0
    df["odds"] = [american_long(min(max(p * factor, LONGSHOT_FLOOR), 0.95)) for p in df["p_title"]]
    return df


def champion(games: pd.DataFrame, season: int) -> str | None:
    """Winner of the NBA Finals for `season`, once the playoffs are over."""
    post = games[(games["season"] == season) & (games["season_type"] == "post-season")]
    if post.empty or (~post["completed"]).any() or post["game_date"].max().month != 6:
        return None
    last = post.sort_values("tipoff_utc").iloc[-1]
    return last["home_team"] if last["home_score"] > last["away_score"] else last["away_team"]
