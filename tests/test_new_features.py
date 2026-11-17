"""Tests for parlays, player props, futures, separate house ledgers, the leaderboard,
database upgrades and the season simulation. No network needed."""

import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src import betting, futures, players
from src import database as db
from src.odds_model import fit_ratings


@pytest.fixture
def tmp_db(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    for u in ("alice", "bob"):
        db.get_or_create_user(u, path)
    return path


def game(gid, home="BOS", away="NY", **kw):
    return {"game_id": gid, "home_team": home, "away_team": away, "completed": False,
            "tipoff_utc": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat(), **kw}


def finals(*rows):
    """rows: (game_id, home, away, home_score, away_score)"""
    return pd.DataFrame([{"game_id": g, "home_team": h, "away_team": a, "home_score": hs, "away_score": as_,
                          "completed": True, "status": "STATUS_FINAL"} for g, h, a, hs, as_ in rows])


# ---------------------------------------------------------------- parlays
def legs():
    return [
        {"game_id": "1", "matchup": "NY @ BOS", "bet_type": "moneyline", "selection": "BOS", "line": None,
         "american_odds": 100},
        {"game_id": "2", "matchup": "MIA @ LAL", "bet_type": "spread", "selection": "LAL", "line": -3.5,
         "american_odds": -110},
    ]


GAMES = {"1": game("1"), "2": game("2", "LAL", "MIA")}


def test_parlay_odds_multiply_and_win(tmp_db):
    pid = betting.place_parlay("alice", legs(), GAMES, 100, db_path=tmp_db)
    p = db.load_parlays("alice", db_path=tmp_db).iloc[0]
    assert p["decimal_odds"] == pytest.approx(2.0 * (1 + 100 / 110))
    assert db.get_balance("alice", tmp_db) == 900
    betting.settle_bets(finals(("1", "BOS", "NY", 110, 100), ("2", "LAL", "MIA", 120, 100)), tmp_db)
    p = db.load_parlays("alice", db_path=tmp_db).iloc[0]
    assert (p["parlay_id"], p["status"]) == (pid, "won")
    assert db.get_balance("alice", tmp_db) == pytest.approx(900 + 100 * 2.0 * (1 + 100 / 110), abs=0.01)


def test_parlay_loses_as_soon_as_one_leg_loses(tmp_db):
    betting.place_parlay("alice", legs(), GAMES, 100, db_path=tmp_db)
    betting.settle_bets(finals(("1", "BOS", "NY", 90, 100)), tmp_db)  # game 2 not played yet
    assert db.load_parlays("alice", db_path=tmp_db).iloc[0]["status"] == "lost"
    assert db.get_balance("alice", tmp_db) == 900


def test_parlay_pushed_leg_drops_out(tmp_db):
    two = legs()
    two[1]["line"] = -20.0
    betting.place_parlay("alice", two, GAMES, 100, db_path=tmp_db)
    betting.settle_bets(finals(("1", "BOS", "NY", 110, 100), ("2", "LAL", "MIA", 120, 100)), tmp_db)  # spread push
    p = db.load_parlays("alice", db_path=tmp_db).iloc[0]
    assert (p["status"], p["payout"]) == ("won", 200.0)  # only the even-money leg counts


@pytest.mark.parametrize("bad,msg", [
    (lambda lg: lg[:1], "2-8 legs"),
    (lambda lg: [lg[0], {**lg[1], "game_id": "1"}], "one leg per game"),
])
def test_parlay_validation(tmp_db, bad, msg):
    with pytest.raises(betting.BetError, match=msg):
        betting.place_parlay("alice", bad(legs()), GAMES, 10, db_path=tmp_db)
    assert db.get_balance("alice", tmp_db) == 1000


# ---------------------------------------------------------------- props
def _box(tmp_db, pts, dnp=False):
    row = {"game_id": "1", "player_id": "77", "season": 2027, "game_date": "2026-11-20", "team": "BOS",
           "opponent": "NY", "is_home": 1, "name": "Test Player", "position": "G", "headshot": None, "starter": 1,
           "dnp": int(dnp), "dnp_reason": None, "minutes": 0 if dnp else 34, "pts": None if dnp else pts,
           "reb": 5, "ast": 4, "fg3m": 2}
    db.upsert_player_games(pd.DataFrame([row]), ["1"], tmp_db)


@pytest.mark.parametrize("pts,dnp,status,balance", [(31, False, "won", 1050), (20, False, "lost", 950),
                                                     (None, True, "void", 1000)])
def test_prop_settlement(tmp_db, pts, dnp, status, balance):
    betting.place_bet("alice", game("1"), "prop", "OVER", 100, 50, line=25.5, player_id="77",
                      player_name="Test Player", prop_stat="pts", db_path=tmp_db)
    f = finals(("1", "BOS", "NY", 110, 100))
    assert betting.settle_bets(f, tmp_db) == 0          # box score not downloaded yet -> wait
    _box(tmp_db, pts, dnp)
    assert betting.settle_bets(f, tmp_db) == 1
    assert db.load_bets("alice", db_path=tmp_db).iloc[0]["status"] == status
    assert db.get_balance("alice", tmp_db) == balance


def test_prop_line_is_half_point_near_recent_average():
    line, mean, sd = players.prop_line(pd.Series([20, 22, 24, 26, 28, 30, 25, 24, 26, 25.0]), "pts")
    assert line % 1 == 0.5 and abs(line - mean) <= 1 and sd >= players.MIN_SD["pts"]


# ---------------------------------------------------------------- futures
def test_future_settles_on_champion(tmp_db):
    betting.place_future("alice", "NY", 2026, 400, 50, db_path=tmp_db)
    betting.place_future("bob", "BOS", 2026, 300, 50, db_path=tmp_db)
    finals_game = pd.DataFrame([{"game_id": "F7", "season": 2026, "season_type": "post-season", "completed": True,
                                 "status": "STATUS_FINAL", "game_date": pd.Timestamp("2026-06-18"),
                                 "tipoff_utc": pd.Timestamp("2026-06-19T00:30Z"), "home_team": "OKC",
                                 "away_team": "NY", "home_score": 99, "away_score": 105}])
    assert futures.champion(finals_game, 2026) == "NY"
    assert betting.settle_bets(finals_game, tmp_db) == 2
    assert db.get_balance("alice", tmp_db) == 1000 - 50 + 250
    assert db.get_balance("bob", tmp_db) == 950


def _synthetic(n_games=600, seed=1):
    rng = np.random.default_rng(seed)
    teams = sorted(futures.EAST | futures.WEST)
    strength = dict(zip(teams, np.linspace(-9, 9, 30), strict=True))
    rows = []
    for i in range(n_games):
        h, a = rng.choice(teams, 2, replace=False)
        m = strength[h] - strength[a] + 2 + rng.normal(0, 12)
        rows.append({"game_id": str(i), "season": 2027, "season_type": "regular-season", "completed": True,
                     "neutral_site": False, "game_date": pd.Timestamp("2026-10-21") + pd.Timedelta(days=i // 10),
                     "home_team": h, "away_team": a, "home_score": 112 + m / 2, "away_score": 112 - m / 2})
    return pd.DataFrame(rows), strength


def test_simulation_is_coherent():
    games, strength = _synthetic()
    sim = futures.simulate(fit_ratings(games), futures.standings(games, 2027), n_sims=2000)
    assert sim["p_title"].sum() == pytest.approx(1)
    assert sim["p_conf"].sum() == pytest.approx(2)
    assert sim["p_playoffs"].sum() == pytest.approx(16)
    best = max(strength, key=strength.get)
    assert sim.iloc[0]["team"] in sorted(strength, key=strength.get)[-4:]  # a top-4 team is the favorite
    assert sim.set_index("team").loc[best, "p_title"] > 0.05
    board_edge, board_fair = futures.futures_odds(sim, True), futures.futures_odds(sim, False)
    assert (board_edge["odds"] <= board_fair["odds"]).all()  # the margin shortens every price


# ---------------------------------------------------------------- ledgers & leaderboard
def test_separate_house_ledgers(tmp_db):
    betting.place_bet("alice", game("1"), "moneyline", "BOS", -110, 110, book="edge", db_path=tmp_db)
    betting.place_bet("bob", game("1"), "moneyline", "NY", 100, 100, book="fair", db_path=tmp_db)
    betting.settle_bets(finals(("1", "BOS", "NY", 110, 100)), tmp_db)   # BOS wins
    edge, fair = db.house_summary("edge", tmp_db), db.house_summary("fair", tmp_db)
    assert edge["house_profit"] == pytest.approx(-100)   # paid alice 210 on 110
    assert fair["house_profit"] == pytest.approx(100)    # kept bob's 100
    tl = db.house_timeline(tmp_db)
    assert set(tl["book"]) == {"edge", "fair"}


def test_leaderboard_ranks_by_net(tmp_db):
    betting.place_bet("alice", game("1"), "moneyline", "BOS", 100, 100, db_path=tmp_db)
    betting.settle_bets(finals(("1", "BOS", "NY", 110, 100)), tmp_db)
    lb = db.leaderboard(tmp_db)
    assert lb.iloc[0]["username"] == "alice" and lb.iloc[0]["net"] == 100 and lb.iloc[0]["rank"] == 1
    assert lb.set_index("username").loc["alice", "wins"] == 1


def test_old_database_is_upgraded(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:  # schema from the first release (no book / prop columns)
        conn.executescript("""
            CREATE TABLE users (username TEXT PRIMARY KEY, balance REAL NOT NULL, created_at TEXT);
            CREATE TABLE bets (bet_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, game_id TEXT NOT NULL,
                matchup TEXT, bet_type TEXT NOT NULL, selection TEXT NOT NULL, line REAL, american_odds INTEGER NOT NULL,
                stake REAL NOT NULL, potential_payout REAL NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                payout REAL DEFAULT 0, placed_at TEXT, settled_at TEXT);
            INSERT INTO users VALUES ('old', 950, '2026-11-16');
            INSERT INTO bets (username, game_id, bet_type, selection, line, american_odds, stake, potential_payout)
                VALUES ('old', '1', 'spread', 'BOS', -3.5, 100, 50, 100);""")
    db.init_db(path)
    bet = db.load_bets("old", db_path=path).iloc[0]
    assert bet["book"] == "fair"  # +100 spread came from fair mode
    assert db.get_balance("old", path) == 950
