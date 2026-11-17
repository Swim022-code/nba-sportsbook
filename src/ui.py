"""Shared Streamlit helpers: cached data loading, sidebar (account, settings, parlay slip),
formatting and LLM streaming."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src import database as db
from src import futures, llm, players
from src.betting import BetError, parlay_odds, place_parlay, settle_bets
from src.data_fetcher import EASTERN, DataFetchError
from src.data_service import load_snapshot_into_db, nba_games, refresh_games
from src.odds_model import TeamRatings, backtest, fit_ratings

CURRENCY = "SimBucks"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,24}$")
STAT_SHORT = {"pts": "PTS", "reb": "REB", "ast": "AST", "fg3m": "3PM"}


# ---------------------------------------------------------------- data
@st.cache_data(ttl=1800, show_spinner="Fetching NBA data from ESPN...")
def _refresh_cached() -> dict:
    return refresh_games()


@st.cache_data(ttl=1800, show_spinner="Updating player box scores...")
def _refresh_players_cached(_version: str) -> int:
    try:
        return players.refresh_player_logs(nba_games(db.load_games()))
    except Exception:  # player data is optional; never block the app on it
        players.load_snapshot_if_empty()
        return 0


@st.cache_data(ttl=1800)
def _load_games_cached(_version: str) -> pd.DataFrame:
    return nba_games(db.load_games())


def get_games() -> pd.DataFrame:
    """Refresh from the API (at most every 30 min) and return all NBA games.

    Falls back to the SQLite cache, then to the bundled CSV snapshots, if ESPN can't be
    reached. Also downloads new box scores and settles any wagers that have been decided.
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
    _refresh_players_cached(info["refreshed_at"])
    n = settle_bets(games)
    if n:
        st.toast(f"🎉 Settled {n} wager(s)! Check **My Bets**.")
    return games


@st.cache_data(ttl=1800)
def get_ratings(games: pd.DataFrame) -> TeamRatings:
    return fit_ratings(games[games["completed"]])


@st.cache_data(ttl=1800)
def get_player_games(_version: str) -> pd.DataFrame:
    return db.load_player_games()


def player_games() -> pd.DataFrame:
    return get_player_games(st.session_state.get("last_refresh", "offline"))


@st.cache_data(ttl=1800, show_spinner="Simulating the rest of the season (5,000 times)...")
def get_title_sim(games: pd.DataFrame, _ratings: TeamRatings, season: int) -> pd.DataFrame:
    return futures.simulate(_ratings, futures.standings(games, season))


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
def fmt_odds(odds) -> str:
    odds = int(odds)
    return f"+{odds}" if odds > 0 else str(odds)


def fmt_line(line: float) -> str:
    return f"+{line:g}" if line > 0 else f"{line:g}"


def money(x: float) -> str:
    return f"{x:,.2f}"


def season_label(season: int) -> str:
    """ESPN labels a season by the year it ends: 2027 -> '2026-27'."""
    return f"{season - 1}-{str(season)[-2:]}"


def describe_wager(w: dict) -> str:
    """Human-readable description of a bet or parlay leg."""
    kind, sel = w["bet_type"], w["selection"]
    odds = fmt_odds(w["american_odds"])
    if kind == "moneyline":
        return f"{sel} ML {odds}"
    if kind == "spread":
        return f"{sel} {fmt_line(w['line'])} ({odds})"
    if kind == "total":
        return f"{sel.title()} {w['line']:g} ({odds})"
    if kind == "prop":
        stat = STAT_SHORT.get(w["prop_stat"], w["prop_stat"])
        return f"{w['player_name']} {sel.title()} {w['line']:g} {stat} ({odds})"
    if kind == "future":
        return f"{sel} to win title ({odds})"
    return f"{sel} ({odds})"


def _fmt_refresh(iso: str) -> str:
    try:
        return pd.Timestamp(iso).tz_convert(EASTERN).strftime("%b %d, %I:%M %p ET")
    except (ValueError, TypeError):
        return iso


def house_edge_on() -> bool:
    """Whether the sportsbook adds its margin (sidebar switch; on by default)."""
    return st.session_state.get("house_edge", True)


def current_book() -> str:
    return "edge" if house_edge_on() else "fair"


# ---------------------------------------------------------------- parlay slip
def parlay_slip() -> list[dict]:
    return st.session_state.setdefault("parlay_slip", [])


def add_to_parlay(leg: dict) -> None:
    """Add a leg (replacing any existing leg from the same game)."""
    slip = [x for x in parlay_slip() if x["game_id"] != leg["game_id"]]
    st.session_state["parlay_slip"] = [*slip, leg]


def _on_edge_toggle() -> None:
    st.session_state["house_edge"] = st.session_state["_house_edge_widget"]
    if st.session_state.get("parlay_slip"):
        st.session_state["parlay_slip"] = []  # prices changed: start a fresh slip
        st.session_state["slip_notice"] = "Odds mode changed, so the parlay slip was cleared."


def _render_parlay_slip(user: str | None, games: pd.DataFrame) -> None:
    slip = parlay_slip()
    st.subheader(f"🧾 Parlay slip ({len(slip)})")
    if note := st.session_state.pop("slip_notice", None):
        st.info(note)
    if not slip:
        st.caption("Use **➕ Add to parlay** on any bet. 2-8 legs, one per game.")
        return
    for i, leg in enumerate(slip):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{describe_wager(leg)}**  \n<small>{leg['matchup']}</small>", unsafe_allow_html=True)
        if c2.button("🗑️", key=f"rm_leg_{i}", help="Remove this leg"):
            slip.pop(i)
            st.rerun()
    dec = parlay_odds(slip)
    american = round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))
    st.markdown(f"**{len(slip)}-leg parlay · {fmt_odds(american)}** (×{dec:.2f})")
    stake = st.number_input(f"Parlay stake ({CURRENCY})", min_value=1.0, value=10.0, step=5.0, key="parlay_stake")
    st.caption(f"Pays **{money(stake * dec)}** if every leg wins")
    c1, c2 = st.columns(2)
    if c1.button("Place parlay", type="primary", disabled=not user or len(slip) < 2, width="stretch"):
        game_map = {g["game_id"]: g for g in games.to_dict("records")}
        try:
            pid = place_parlay(user, slip, game_map, stake, book=current_book(), now=datetime.now(timezone.utc))
            st.session_state["parlay_slip"] = []
            st.session_state["bet_confirmation"] = (
                f"Parlay #{pid} placed: {len(slip)} legs for {money(stake)} {CURRENCY}")
            st.rerun()
        except BetError as exc:
            st.error(str(exc))
    if c2.button("Clear", width="stretch"):
        st.session_state["parlay_slip"] = []
        st.rerun()
    if len(slip) < 2:
        st.caption("Add at least one more leg.")
    elif not user:
        st.caption("Log in to place a parlay.")


# ---------------------------------------------------------------- sidebar
def _login(name: str) -> None:
    db.init_db()
    db.get_or_create_user(name)
    st.session_state["user"] = name
    st.query_params["user"] = name  # keeps you logged in if the page is refreshed


def sidebar(games: pd.DataFrame | None = None, show_slip: bool = False) -> str | None:
    """Login box, balance, optional parlay slip and settings. Returns the username (or None)."""
    if "user" not in st.session_state and USERNAME_RE.match(st.query_params.get("user", "")):
        _login(st.query_params["user"])

    with st.sidebar:
        st.markdown('<div class="sb-brand"><span>🏀</span>NBA Mock Sportsbook</div>', unsafe_allow_html=True)
        st.caption("Fake money only. Nothing real is wagered.")
        user = st.session_state.get("user")
        if user:
            st.metric(f"Balance ({CURRENCY})", money(db.get_balance(user)))
            c1, c2 = st.columns([3, 2])
            c1.caption(f"Logged in as **{user}**")
            if c2.button("Log out", width="stretch"):
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

        if show_slip and games is not None:
            st.divider()
            _render_parlay_slip(user, games)

        st.divider()
        st.subheader("Settings")
        # Persist across pages: copy the saved preference into the widget's key before drawing it.
        st.session_state["_house_edge_widget"] = house_edge_on()
        st.toggle("🏦 House edge", key="_house_edge_widget", on_change=_on_edge_toggle,
                  help="ON: realistic book. Spreads/totals/props at -110, moneylines ~4.5% margin, futures ~25%. "
                       "OFF: fair odds with zero expected profit for either side. Each mode has its own house "
                       "ledger (see Leaderboard). Applies to new bets.")
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
        if st.button("🔄 Refresh data", help="Re-download the latest scores, schedule and box scores from ESPN"):
            for fn in (_refresh_cached, _refresh_players_cached, _load_games_cached, get_ratings,
                       get_player_games, get_title_sim):
                fn.clear()
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
