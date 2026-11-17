"""Shared Streamlit helpers: cached data loading, sidebar, formatting."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from src import database as db
from src import llm
from src.betting import settle_bets
from src.data_fetcher import EASTERN, DataFetchError
from src.data_service import load_snapshot_into_db, nba_games, refresh_games
from src.odds_model import TeamRatings, backtest, fit_ratings

CURRENCY = "SimBucks"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,24}$")


# ---------------------------------------------------------------- data
@st.cache_data(ttl=1800, show_spinner="Fetching NBA data from ESPN...")
def _refresh_cached() -> dict:
    return refresh_games()


@st.cache_data(ttl=1800)
def _load_games_cached(_version: str) -> pd.DataFrame:
    return nba_games(db.load_games())


def get_games() -> pd.DataFrame:
    """Refresh from the API (at most every 30 min) and return all NBA games.

    Falls back to the SQLite cache, then to the bundled CSV snapshot, if ESPN
    can't be reached. Also settles any bets whose games have finished.
    """
    db.init_db()
    try:
        info = _refresh_cached()
        st.session_state["last_refresh"] = info["refreshed_at"]
        st.session_state.pop("offline", None)
    except DataFetchError as exc:
        st.session_state["offline"] = str(exc)
        if db.load_games().empty:
            load_snapshot_into_db()  # first run with no internet: use the bundled CSV
        info = {"refreshed_at": "offline"}
    games = _load_games_cached(info["refreshed_at"])
    if games.empty:
        st.error("No game data available. Check your internet connection, then click **🔄 Refresh data**.")
        st.stop()
    n = settle_bets(games)
    if n:
        st.toast(f"🎉 Settled {n} bet(s)! Check **My Bets**.")
    return games


@st.cache_data(ttl=1800)
def get_ratings(games: pd.DataFrame) -> TeamRatings:
    return fit_ratings(games[games["completed"]])


@st.cache_data(ttl=3600, show_spinner="Back-testing the odds model (about 10 s)...")
def get_backtest(games: pd.DataFrame) -> pd.DataFrame:
    return backtest(games[games["completed"]])


def backtest_summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {}
    return {
        "games_tested": len(bt),
        "accuracy": round(float(((bt["p_home"] > 0.5) == bt["home_won"]).mean()), 3),
        "always_pick_home_accuracy": round(float(bt["home_won"].mean()), 3),
        "brier": round(float(((bt["p_home"] - bt["home_won"]) ** 2).mean()), 3),
        "margin_mae": round(float((bt["pred_margin"] - bt["actual_margin"]).abs().mean()), 2),
        "total_mae": round(float((bt["pred_total"] - bt["actual_total"]).abs().mean()), 2),
    }


# ---------------------------------------------------------------- formatting
def fmt_odds(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


def fmt_line(line: float) -> str:
    return f"+{line:g}" if line > 0 else f"{line:g}"


def money(x: float) -> str:
    return f"{x:,.2f}"


def season_label(season: int) -> str:
    """ESPN labels a season by the year it ends: 2027 -> '2026-27'."""
    return f"{season - 1}-{str(season)[-2:]}"


def _fmt_refresh(iso: str) -> str:
    try:
        return pd.Timestamp(iso).tz_convert(EASTERN).strftime("%b %d, %I:%M %p ET")
    except (ValueError, TypeError):
        return iso


def house_edge_on() -> bool:
    """Whether the sportsbook adds its margin (sidebar switch; on by default)."""
    return st.session_state.get("house_edge", True)


# ---------------------------------------------------------------- sidebar
def _login(name: str) -> None:
    db.init_db()
    db.get_or_create_user(name)
    st.session_state["user"] = name
    st.query_params["user"] = name  # keeps you logged in if the page is refreshed


def sidebar() -> str | None:
    """Login box, balance and settings. Returns the logged-in username (or None)."""
    # Restore the login from the URL after a browser refresh.
    if "user" not in st.session_state and USERNAME_RE.match(st.query_params.get("user", "")):
        _login(st.query_params["user"])

    with st.sidebar:
        st.markdown('<div class="sb-brand"><span>🏀</span>NBA Mock Sportsbook</div>', unsafe_allow_html=True)
        st.caption("Fake money only. Nothing real is wagered.")
        st.subheader("Account")
        user = st.session_state.get("user")
        if user:
            st.metric(f"Balance ({CURRENCY})", money(db.get_balance(user)))
            st.caption(f"Logged in as **{user}**")
            if st.button("Log out"):
                st.session_state.pop("user")
                st.query_params.pop("user", None)
                st.rerun()
        else:
            with st.form("login"):
                name = st.text_input("Pick a username", max_chars=24,
                                     help=f"Letters, numbers, _ . - only. New users start with "
                                          f"{db.STARTING_BALANCE:,.0f} {CURRENCY}.")
                if st.form_submit_button("Log in / Sign up", type="primary"):
                    if USERNAME_RE.match(name.strip()):
                        _login(name.strip())
                        st.rerun()
                    else:
                        st.error("Use 2-24 letters, numbers, _ . or -")

        st.divider()
        st.subheader("Settings")
        # Persist across pages: copy the saved preference into the widget's key before drawing it.
        st.session_state["_house_edge_widget"] = house_edge_on()
        st.toggle("🏦 House edge", key="_house_edge_widget",
                  on_change=lambda: st.session_state.update(house_edge=st.session_state["_house_edge_widget"]),
                  help="ON: realistic book. Spreads/totals at -110 and moneylines with a ~4.5% margin, "
                       "so the house profits over time. OFF: fair odds (even money, true probabilities): "
                       "zero expected profit for either side. Applies to new bets.")
        options = [*llm.MODEL_OPTIONS, "Other..."]
        current = st.session_state.get("model", llm.DEFAULT_MODEL)
        choice = st.selectbox("Claude model", options,
                              index=options.index(current) if current in options else len(options) - 1,
                              help="Model used by the AI features.")
        if choice == "Other...":
            choice = st.text_input("Model ID", value=current if current not in options else "")
        st.session_state["model"] = choice or llm.DEFAULT_MODEL

        if llm.is_configured():
            st.success("Anthropic API key found", icon="✅")
        else:
            st.warning("No Anthropic API key: AI features are off. See README → Setup.", icon="🔑")
        if st.button("🔄 Refresh data", help="Re-download the latest scores and schedule from ESPN"):
            _refresh_cached.clear()
            _load_games_cached.clear()
            get_ratings.clear()
            st.rerun()
        if st.session_state.get("offline"):
            st.warning("ESPN unreachable: showing saved data.", icon="📴")
        elif "last_refresh" in st.session_state:
            st.caption(f"Data updated {_fmt_refresh(st.session_state['last_refresh'])}")
        st.caption("Data: ESPN public NBA API · Fake currency only, no real money")
    return st.session_state.get("user")


# ---------------------------------------------------------------- LLM
def stream_llm(generator_fn, *args, **kwargs) -> str | None:
    """Run an llm.* generator, streaming text into the page, with friendly errors."""
    if not llm.is_configured():
        st.info("Add an `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` to enable AI features "
                "(see README → Setup).", icon="🔑")
        return None
    try:
        with st.spinner("Claude is thinking..."):
            gen = generator_fn(*args, model=st.session_state.get("model", llm.DEFAULT_MODEL), **kwargs)
            first = next(gen, "")  # wait for the first token inside the spinner
        return st.write_stream(_prepend(first, gen))
    except llm.LLMError as exc:
        st.error(str(exc), icon="⚠️")
        return None


def _prepend(first: str, rest):
    if first:
        yield first
    yield from rest
