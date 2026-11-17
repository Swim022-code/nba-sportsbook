"""Unit tests for odds math, bet grading/settlement, parsing and EDA helpers.

Run:  pytest -q
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src import betting, database as db, eda
from src.data_fetcher import parse_events
from src.odds_model import (american_to_decimal, fit_ratings, make_lines, payout_for,
                            prob_to_american)


# ---------------------------------------------------------------- odds math
def test_american_odds_conversions():
    assert american_to_decimal(-110) == pytest.approx(1.9091, rel=1e-3)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert payout_for(100, -200) == 150.0
    assert payout_for(100, 130) == 230.0


def test_prob_to_american():
    assert prob_to_american(0.5) in (-100, 100)
    assert prob_to_american(2 / 3) == -200
    assert prob_to_american(0.25) == 300


# ---------------------------------------------------------------- grading
@pytest.mark.parametrize("bet,expected", [
    ({"bet_type": "moneyline", "selection": "BOS", "line": None}, "won"),
    ({"bet_type": "moneyline", "selection": "NY", "line": None}, "lost"),
    ({"bet_type": "spread", "selection": "BOS", "line": -5.5}, "won"),    # won by 6
    ({"bet_type": "spread", "selection": "BOS", "line": -6.0}, "push"),
    ({"bet_type": "spread", "selection": "NY", "line": 5.5}, "lost"),
    ({"bet_type": "spread", "selection": "NY", "line": 6.5}, "won"),
    ({"bet_type": "total", "selection": "OVER", "line": 210.5}, "won"),   # total 214
    ({"bet_type": "total", "selection": "UNDER", "line": 214.0}, "push"),
    ({"bet_type": "total", "selection": "UNDER", "line": 220.5}, "won"),
])
def test_grade_bet(bet, expected):
    assert betting.grade_bet(bet, home_team="BOS", home_score=110, away_score=104) == expected


# ---------------------------------------------------------------- placement + settlement
@pytest.fixture
def tmp_db(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    db.get_or_create_user("alice", path)
    return path


def _game(**kw):
    base = {"game_id": "1", "home_team": "BOS", "away_team": "NY", "completed": False,
            "tipoff_utc": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()}
    return {**base, **kw}


def test_place_and_settle_winning_bet(tmp_db):
    betting.place_bet("alice", _game(), "moneyline", "BOS", -150, 150, db_path=tmp_db)
    assert db.get_balance("alice", tmp_db) == 850
    final = pd.DataFrame([{"game_id": "1", "completed": True, "home_team": "BOS",
                           "home_score": 100.0, "away_score": 90.0}])
    assert betting.settle_bets(final, tmp_db) == 1
    assert db.get_balance("alice", tmp_db) == pytest.approx(1100)  # 850 + 150 * 1.6667
    assert betting.settle_bets(final, tmp_db) == 0                 # never settles twice


def test_push_refunds_stake(tmp_db):
    betting.place_bet("alice", _game(), "total", "OVER", -110, 100, line=190.0, db_path=tmp_db)
    final = pd.DataFrame([{"game_id": "1", "completed": True, "home_team": "BOS",
                           "home_score": 100.0, "away_score": 90.0}])
    betting.settle_bets(final, tmp_db)
    assert db.get_balance("alice", tmp_db) == 1000


@pytest.mark.parametrize("kwargs,msg", [
    ({"stake": 0}, "greater than 0"),
    ({"stake": 5000}, "Insufficient"),
    ({"selection": "LAL"}, "Selection"),
    ({"game": _game(tipoff_utc="2020-01-01T00:00:00+00:00")}, "closed"),
    ({"game": _game(completed=True)}, "final"),
])
def test_invalid_bets_rejected(tmp_db, kwargs, msg):
    args = {"game": _game(), "bet_type": "moneyline", "selection": "BOS", "american_odds": -110, "stake": 10}
    args.update(kwargs)
    with pytest.raises(betting.BetError, match=msg):
        betting.place_bet("alice", db_path=tmp_db, **args)
    assert db.get_balance("alice", tmp_db) == 1000


# ---------------------------------------------------------------- parsing
def test_parse_events_flattens_one_row_per_game():
    payload = {"events": [{
        "id": "9", "date": "2026-11-16T00:30Z", "season": {"year": 2027, "slug": "regular-season"},
        "status": {"type": {"name": "STATUS_FINAL", "completed": True}},
        "competitions": [{"neutralSite": False, "venue": {"fullName": "Arena"}, "competitors": [
            {"homeAway": "home", "score": "101", "team": {"abbreviation": "BOS", "displayName": "Boston"},
             "statistics": [{"name": "rebounds", "displayValue": "44"}], "records": [{"type": "total", "summary": "5-1"}]},
            {"homeAway": "away", "score": "99", "team": {"abbreviation": "NY", "displayName": "New York"}},
        ]}],
    }]}
    rows = parse_events(payload)
    assert len(rows) == 1
    r = rows[0]
    assert (r["home_team"], r["home_score"], r["home_reb"], r["away_reb"]) == ("BOS", 101.0, 44.0, None)
    assert r["game_date"] == "2026-11-15"  # 7:30pm Eastern


# ---------------------------------------------------------------- model + EDA
def _synthetic_season(n_games=400, seed=0):
    """Games generated from known team strengths, to check the model recovers them."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    strength = dict(zip(teams, np.linspace(-8, 8, 10)))
    rows = []
    for i in range(n_games):
        h, a = rng.choice(teams, 2, replace=False)
        margin = strength[h] - strength[a] + 3 + rng.normal(0, 11)
        total = 225 + rng.normal(0, 15)
        rows.append({"game_id": str(i), "game_date": pd.Timestamp("2026-10-20") + pd.Timedelta(days=i // 8),
                     "season": 2027, "season_type": "regular-season", "completed": True, "neutral_site": False,
                     "home_team": h, "away_team": a,
                     "home_score": round((total + margin) / 2), "away_score": round((total - margin) / 2)})
    return pd.DataFrame(rows), strength


def test_ratings_recover_known_strengths():
    games, strength = _synthetic_season()
    r = fit_ratings(games)
    true = pd.Series(strength).reindex(r.ratings.index)
    assert r.ratings.rank().corr(true.rank()) > 0.85   # ranking is recovered
    assert 1.0 < r.home_court < 5.0
    lines = make_lines(r, "T9", "T0")
    assert lines["home_ml"] < 0 < lines["away_ml"] and lines["home_spread"] < 0


def test_iqr_outliers():
    s = pd.Series([10, 11, 12, 11, 10, 50])
    assert eda.iqr_outliers(s).tolist() == [False] * 5 + [True]
