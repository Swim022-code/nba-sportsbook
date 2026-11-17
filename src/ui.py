"""Shared Streamlit helpers: cached data loading, sidebar, formatting."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import database as db
from src import llm
from src.betting import settle_bets
from src.data_fetcher import DataFetchError
from src.data_service import load_snapshot_into_db, nba_games, refresh_games
from src.odds_model import fit_ratings

CURRENCY = "SimBucks"


@st.cache_data(ttl=1800, show_spinner="Fetching NBA data from ESPN...")
def _refresh_cached() -> dict:
    return refresh_games()


@st.cache_data(ttl=1800)
def _load_games_cached(_version: str) -> pd.DataFrame:
    return nba_games(db.load_games())


def get_games() -> pd.DataFrame:
    """Refresh from the API (at most every 30 min) and return all NBA games."""
    db.init_db()
    try:
        info = _refresh_cached()
        st.session_state["last_refresh"] = info["refreshed_at"]
    except DataFetchError as exc:
        st.warning(f"Live data unavailable, using cached data. ({exc})")
        if db.load_games().empty:
            load_snapshot_into_db()  # first run with no internet: use the bundled CSV
        info = {"refreshed_at": "cache"}
    games = _load_games_cached(info["refreshed_at"])
    if games.empty:
        st.error("No game data available. Check your internet connection and use **Refresh data**.")
        st.stop()
    # Grade any bets whose games just went final.
    n = settle_bets(games)
    if n:
        st.toast(f"Settled {n} bet(s)!")
    return games


@st.cache_data(ttl=1800)
def get_ratings(games: pd.DataFrame):
    return fit_ratings(games[games["completed"]])


def fmt_odds(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


def fmt_line(line: float) -> str:
    return f"+{line:g}" if line > 0 else f"{line:g}"


def money(x: float) -> str:
    return f"{x:,.2f}"


def sidebar() -> str | None:
    """Login box, balance and settings. Returns the logged-in username (or None)."""
    with st.sidebar:
        st.header("🏀 Account")
        user = st.session_state.get("user")
        if user:
            st.metric(f"Balance ({CURRENCY})", money(db.get_balance(user)))
            st.caption(f"Logged in as **{user}**")
            if st.button("Log out"):
                st.session_state.pop("user")
                st.rerun()
        else:
            with st.form("login"):
                name = st.text_input("Pick a username", max_chars=24,
                                     help=f"New users start with {db.STARTING_BALANCE:,.0f} {CURRENCY}.")
                if st.form_submit_button("Log in / Sign up") and name.strip():
                    db.init_db()
                    db.get_or_create_user(name)
                    st.session_state["user"] = name.strip()
                    st.rerun()

        st.divider()
        st.header("⚙️ Settings")
        st.session_state["model"] = st.text_input(
            "Claude model", value=st.session_state.get("model", llm.DEFAULT_MODEL),
            help="Model ID used for the AI features.")
        if llm.is_configured():
            st.success("AI features: connected", icon="✅")
        else:
            st.warning("AI features: no API key. See README → Setup.", icon="🔑")
        if st.button("🔄 Refresh data"):
            _refresh_cached.clear()
            _load_games_cached.clear()
            get_ratings.clear()
            st.rerun()
        if "last_refresh" in st.session_state:
            st.caption(f"Data refreshed {st.session_state['last_refresh']} UTC")
        st.caption("Data: ESPN public NBA API · Fake currency only")
    return st.session_state.get("user")


def stream_llm(generator_fn, *args, **kwargs) -> str | None:
    """Run an llm.* generator, streaming into the page, with friendly errors."""
    if not llm.is_configured():
        st.info("Add an `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` to enable AI features "
                "(see README).", icon="🔑")
        return None
    try:
        return st.write_stream(generator_fn(*args, model=st.session_state.get("model", llm.DEFAULT_MODEL), **kwargs))
    except llm.LLMError as exc:
        st.error(str(exc))
        return None
