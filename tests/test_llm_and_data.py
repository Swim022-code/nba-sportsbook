"""Tests for the LLM module (with a fake Anthropic client - no API key or network
needed), voided bets, the offline snapshot and the newer EDA helpers."""

import json
from datetime import datetime, timedelta, timezone

import anthropic

try:  # newer anthropic SDKs use httpx2, older ones httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx
import pandas as pd
import pytest

from src import betting, data_service, eda, llm
from src import database as db


# ---------------------------------------------------------------- fake Anthropic client
class FakeStream:
    def __init__(self, chunks, stop_reason="end_turn"):
        self.text_stream = iter(chunks)
        self._stop = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return type("Msg", (), {"stop_reason": self._stop})()


def _api_error(cls, status):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(message="boom", response=httpx.Response(status, request=req), body=None)


class FakeClient:
    """Records each call; `plan` is a list of FakeStream or exceptions, one per call."""

    def __init__(self, plan):
        self.plan, self.calls = list(plan), []
        self.messages = self.beta = self
        self.beta.messages = self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        item = self.plan.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake(monkeypatch):
    def install(plan):
        client = FakeClient(plan)
        monkeypatch.setattr(llm, "_client", lambda: client)
        return client
    return install


def test_stream_returns_text_and_grounds_prompt(fake):
    client = fake([FakeStream(["Hello ", "world"])])
    out = "".join(llm.write_eda_report({"home_win_pct": 0.55}))
    assert out == "Hello world"
    call = client.calls[0]
    assert call["model"] == llm.DEFAULT_MODEL
    assert "0.55" in call["messages"][0]["content"]          # our data is in the prompt
    assert "Ground every number" in call["system"]


def test_bad_request_retries_in_basic_mode(fake):
    client = fake([_api_error(anthropic.BadRequestError, 400), FakeStream(["ok"])])
    assert "".join(llm.chat("hi", [], {"x": 1})) == "ok"
    assert "betas" in client.calls[0] and "betas" not in client.calls[1]


@pytest.mark.parametrize("exc,status,msg", [
    (anthropic.AuthenticationError, 401, "key was rejected"),
    (anthropic.NotFoundError, 404, "not found"),
    (anthropic.RateLimitError, 429, "Rate limited"),
    (anthropic.InternalServerError, 500, r"API error \(500\)"),
])
def test_api_errors_become_friendly_messages(fake, exc, status, msg):
    fake([_api_error(exc, status)])
    with pytest.raises(llm.LLMError, match=msg):
        "".join(llm.game_preview({"game": "A @ B"}))


def test_refusal_and_truncation_are_reported(fake):
    fake([FakeStream(["partial"], stop_reason="max_tokens")])
    assert "truncated" in "".join(llm.explain_findings("x", {}))
    fake([FakeStream([], stop_reason="refusal")])
    assert "declined" in "".join(llm.explain_findings("x", {}))


def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "get_api_key", lambda: None)
    with pytest.raises(llm.LLMError, match="No ANTHROPIC_API_KEY"):
        "".join(llm.write_eda_report({}))


def test_chat_keeps_history_and_attaches_context(fake):
    client = fake([FakeStream(["a"])])
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    "".join(llm.chat("q2", history, {"balance": 900}))
    msgs = client.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert json.dumps({"balance": 900}, separators=(",", ":")) in msgs[-1]["content"]  # compact JSON


# ---------------------------------------------------------------- void bets
def test_postponed_game_voids_and_refunds(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    db.get_or_create_user("bob", path)
    game = {"game_id": "7", "home_team": "BOS", "away_team": "NY", "completed": False,
            "tipoff_utc": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()}
    betting.place_bet("bob", game, "moneyline", "NY", 150, 200, db_path=path)
    assert db.get_balance("bob", path) == 800
    games = pd.DataFrame([{**game, "status": "STATUS_POSTPONED", "home_score": None, "away_score": None}])
    assert betting.settle_bets(games, path) == 1
    assert db.get_balance("bob", path) == 1000
    assert db.load_bets("bob", db_path=path)["status"].tolist() == ["void"]


# ---------------------------------------------------------------- data pipeline
def test_snapshot_loads_offline(tmp_path):
    n = data_service.load_snapshot_into_db(tmp_path / "s.db")
    games = data_service.nba_games(db.load_games(tmp_path / "s.db"))
    assert n > 1000
    teams = set(games["home_team"]) | set(games["away_team"])
    assert len(teams) == 30                                   # All-Star "teams" removed
    assert games.loc[games["completed"], "home_score"].notna().all()


def test_month_keys_cross_year():
    from datetime import date

    from src.data_fetcher import month_keys
    assert month_keys(date(2025, 11, 5), date(2026, 2, 1)) == ["202511", "202512", "202601", "202602"]


def test_scoring_trend_and_correlation_text():
    games = pd.DataFrame({
        "game_id": ["1", "2", "3"], "completed": [True] * 3,
        "game_date": pd.to_datetime(["2026-01-05", "2026-01-20", "2026-02-03"]),
        "home_score": [110, 100, 120], "away_score": [100, 105, 90],
    })
    trend = eda.scoring_trend(games)
    assert trend["games"].tolist() == [2, 1]
    assert trend["avg_total"].tolist() == [207.5, 210.0]
    assert eda.scoring_trend(games, min_games=2)["games"].tolist() == [2]   # noisy month dropped
    text = eda.describe_correlations(pd.Series({"won": 1.0, "fg_pct": 0.5, "fg3a": -0.01, "reb": 0.3}))
    assert "FG %" in text and "3PT attempted" in text
