"""Sportsbook: upcoming games with game lines, player props, parlays and AI previews."""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src import llm, players, style
from src.betting import VOID_STATUSES, BetError, place_bet
from src.data_fetcher import EASTERN
from src.odds_model import american_to_implied_prob, book_margin, make_lines, model_probability, payout_for
from src.ui import (
    CURRENCY,
    STAT_SHORT,
    add_to_parlay,
    current_book,
    fmt_line,
    fmt_odds,
    get_games,
    get_ratings,
    house_edge_on,
    money,
    player_games,
    sidebar,
    stream_llm,
)

games = get_games()
user = sidebar(games, show_slip=True)
ratings = get_ratings(games)
pg = player_games()
edge = house_edge_on()

style.hero("Sportsbook", "Game lines, player props and parlays · odds from our model · bets settle automatically",
           "🏀", style.edge_badge(edge))

if msg := st.session_state.pop("bet_confirmation", None):
    st.toast(msg, icon="✅")
    st.success(msg + "  ·  Track it in **🎟️ My Bets**.", icon="✅")

now = pd.Timestamp(datetime.now(timezone.utc))
upcoming = games[~games["completed"] & ~games["status"].isin(VOID_STATUSES)].sort_values("tipoff_utc").copy()
if upcoming.empty:
    st.info("No upcoming games in the next 7 days (off-season?). Try **🏆 Futures** instead.")
    st.stop()
upcoming["local_date"] = upcoming["tipoff_utc"].dt.tz_convert(EASTERN).dt.date

f1, f2 = st.columns([1, 2])
days = sorted(upcoming["local_date"].unique())
open_days = sorted(upcoming.loc[upcoming["tipoff_utc"] > now, "local_date"].unique())
day = f1.selectbox("Game day (ET)", days, index=days.index(open_days[0]) if open_days else 0,
                   format_func=lambda d: d.strftime("%a %b %d"))
teams = sorted(set(upcoming["home_team"]) | set(upcoming["away_team"]))
team_filter = f2.multiselect("Filter by team", teams, placeholder="All teams")

slate = upcoming[upcoming["local_date"] == day]
if team_filter:
    slate = slate[slate["home_team"].isin(team_filter) | slate["away_team"].isin(team_filter)]
if slate.empty:
    st.info("No games match those filters.")
    st.stop()


def matchup_context(g, lines) -> dict:
    """Data handed to the LLM for a game preview."""
    def team(t):
        return {"net_rating": round(float(ratings.ratings.get(t, 0)), 2),
                "offense_pts": round(float(ratings.offense.get(t, 0)), 1),
                "defense_pts_allowed": round(float(ratings.defense.get(t, 0)), 1),
                "rating_rank": int(list(ratings.ratings.index).index(t) + 1) if t in ratings.ratings else None}
    return {
        "matchup": f"{g.away_team_name} ({g.away_record}) at {g.home_team_name} ({g.home_record})",
        "tipoff_et": g.tipoff_utc.tz_convert(EASTERN).strftime("%a %b %d %I:%M %p"),
        "home": {"team": g.home_team, **team(g.home_team)},
        "away": {"team": g.away_team, **team(g.away_team)},
        "model": {
            "home_court_advantage_pts": round(ratings.home_court, 2),
            "predicted_home_margin": round(float(lines["pred_margin"]), 1),
            "predicted_total": round(float(lines["pred_total"]), 1),
            "home_win_probability": round(lines["p_home"], 3),
            "margin_std_dev": round(ratings.sigma, 1),
        },
        "posted_lines": {
            "house_edge": "on (~4.5% margin)" if lines["house_edge"] else "off (fair odds)",
            "moneyline": {g.home_team: lines["home_ml"], g.away_team: lines["away_ml"]},
            "moneyline_implied_prob": {g.home_team: round(american_to_implied_prob(lines["home_ml"]), 3),
                                       g.away_team: round(american_to_implied_prob(lines["away_ml"]), 3)},
            "spread": {g.home_team: lines["home_spread"], g.away_team: lines["away_spread"],
                       "price": lines["spread_odds"]},
            "total": {"line": lines["total"], "price": lines["total_odds"]},
        },
    }


def ev_of(p_win: float, odds: int, stake: float) -> float:
    payout = payout_for(stake, odds)
    return p_win * (payout - stake) - (1 - p_win) * stake


def place_or_add(g, leg: dict, stake: float, key: str) -> None:
    """'Place bet' and '➕ Add to parlay' buttons for one selection."""
    c1, c2 = st.columns([3, 2])
    label = leg["label"]
    if c1.button(f"Place bet: {label}", type="primary", key=f"place_{key}", width="stretch", disabled=not user):
        try:
            bet_id = place_bet(user, g._asdict(), leg["bet_type"], leg["selection"], leg["american_odds"], stake,
                               leg.get("line"), book=current_book(), player_id=leg.get("player_id"),
                               player_name=leg.get("player_name"), prop_stat=leg.get("prop_stat"))
            st.session_state["bet_confirmation"] = (
                f"Bet #{bet_id} placed: **{label}** ({g.away_team} @ {g.home_team}) for {money(stake)} {CURRENCY}")
            st.rerun()  # refresh so the sidebar balance updates immediately
        except BetError as exc:
            st.error(str(exc))
    if c2.button("➕ Add to parlay", key=f"parlay_{key}", width="stretch"):
        add_to_parlay({k: v for k, v in leg.items() if k != "label"} | {"label": label})
        st.toast(f"Added to parlay: {label}", icon="🧾")
        st.rerun()
    if not user:
        st.caption("Log in (sidebar) to place bets.")


def game_lines_tab(g, lines) -> None:
    gid = g.game_id
    market = st.radio("Market", ["moneyline", "spread", "total"], horizontal=True, key=f"mkt_{gid}",
                      format_func=str.title)
    if market == "moneyline":
        opts = [(g.away_team, lines["away_ml"], None, "away"), (g.home_team, lines["home_ml"], None, "home")]
        labels = [f"{t} {fmt_odds(o)}" for t, o, _, _ in opts]
    elif market == "spread":
        o = lines["spread_odds"]
        opts = [(g.away_team, o, lines["away_spread"], "away"), (g.home_team, o, lines["home_spread"], "home")]
        labels = [f"{t} {fmt_line(ln)} ({fmt_odds(o)})" for t, o, ln, _ in opts]
    else:
        o = lines["total_odds"]
        opts = [("OVER", o, lines["total"], "over"), ("UNDER", o, lines["total"], "under")]
        labels = [f"{s.title()} {ln:g} ({fmt_odds(o)})" for s, o, ln, _ in opts]
    i = st.radio("Pick", range(2), format_func=lambda k: labels[k], key=f"sel_{gid}_{market}_{edge}")
    sel, odds, line, side = opts[i]
    stake = st.number_input(f"Stake ({CURRENCY})", min_value=1.0, value=50.0, step=10.0, key=f"stake_{gid}")
    p_model = model_probability(ratings, lines, market, side)
    payout = payout_for(stake, odds)
    style.slip_summary(payout, payout - stake, p_model, american_to_implied_prob(odds), ev_of(p_model, odds, stake))
    leg = {"game_id": gid, "matchup": f"{g.away_team} @ {g.home_team}", "bet_type": market, "selection": sel,
           "line": line, "american_odds": odds, "label": labels[i]}
    place_or_add(g, leg, stake, f"{gid}_lines")


def props_tab(g) -> None:
    gid = g.game_id
    menu = players.prop_menu(pg, g.away_team, g.home_team, edge) + players.prop_menu(pg, g.home_team, g.away_team, edge)
    if not menu:
        st.info("No player props for this game yet (not enough recent box scores).")
        return
    by_player: dict[str, list[dict]] = {}
    for m in menu:
        by_player.setdefault(m["player_id"], []).append(m)
    pids = list(by_player)
    c1, c2 = st.columns([3, 2])
    pid = c1.selectbox("Player", pids, key=f"prop_player_{gid}",
                       format_func=lambda x: f"{by_player[x][0]['player_name']} ({by_player[x][0]['team']})")
    stats = {m["stat"]: m for m in by_player[pid]}
    stat = c2.radio("Stat", list(stats), horizontal=True, key=f"prop_stat_{gid}_{pid}",
                    format_func=lambda s: STAT_SHORT[s])
    m = stats[stat]
    recent = players.played(pg[pg["player_id"] == pid]).sort_values("game_date").tail(10)
    hit = (recent[stat] > m["line"]).mean() if len(recent) else 0.0
    style.player_card(m["player_name"], m["team"], m["position"], m["headshot"], {
        "LINE": f"{m['line']:g} {STAT_SHORT[stat]}", "LAST 10 AVG": f"{recent[stat].mean():.1f}",
        "OVER IN LAST 10": f"{hit:.0%}", "MODEL P(OVER)": f"{m['p_over']:.0%}"})
    st.write("")
    side = st.radio("Pick", ["OVER", "UNDER"], horizontal=True, key=f"prop_side_{gid}_{pid}_{stat}",
                    format_func=lambda s: f"{s.title()} {m['line']:g} ({fmt_odds(m['over_odds'])})")
    odds = m["over_odds"] if side == "OVER" else m["under_odds"]
    stake = st.number_input(f"Stake ({CURRENCY})", min_value=1.0, value=25.0, step=5.0, key=f"prop_stake_{gid}")
    p_win = m["p_over"] if side == "OVER" else 1 - m["p_over"]
    payout = payout_for(stake, odds)
    style.slip_summary(payout, payout - stake, p_win, american_to_implied_prob(odds), ev_of(p_win, odds, stake))
    label = f"{m['player_name']} {side.title()} {m['line']:g} {STAT_SHORT[stat]} ({fmt_odds(odds)})"
    leg = {"game_id": gid, "matchup": f"{g.away_team} @ {g.home_team}", "bet_type": "prop", "selection": side,
           "line": m["line"], "american_odds": odds, "player_id": pid, "player_name": m["player_name"],
           "prop_stat": stat, "label": label}
    place_or_add(g, leg, stake, f"{gid}_prop")
    st.caption("Lines are each player's recency-weighted average (last 20 games). A player who doesn't play → void.")


def render_game(g) -> None:
    lines = make_lines(ratings, g.home_team, g.away_team, g.neutral_site, house_edge=edge)
    started = g.tipoff_utc <= now
    tip = g.tipoff_utc.tz_convert(EASTERN).strftime("%a %I:%M %p ET")
    status = ('<span class="sb-live">● LIVE · betting closed</span>' if started
              else f'<span class="sb-time">🕒 {tip}</span>')
    margin = book_margin(lines["home_ml"], lines["away_ml"])
    foot = (f"Projected score: {g.away_team} {lines['pred_away_pts']:.0f}, {g.home_team} {lines['pred_home_pts']:.0f}"
            f" · Moneyline house margin {margin:.1%}")
    style.game_card(g, lines, status, foot)
    if not started:
        with st.expander("🎟️ Bet this game: lines, player props, AI preview"):
            t1, t2, t3 = st.tabs(["Game lines", "🎯 Player props", "🤖 AI preview"])
            with t1:
                game_lines_tab(g, lines)
            with t2:
                props_tab(g)
            with t3:
                if st.button("Generate preview", key=f"ai_{g.game_id}", disabled=not llm.is_configured()):
                    stream_llm(llm.game_preview, matchup_context(g, lines))
                elif not llm.is_configured():
                    st.caption("Requires an Anthropic API key (see README).")
    st.write("")


open_games = slate[slate["tipoff_utc"] > now]
live_games = slate[slate["tipoff_utc"] <= now]
if open_games.empty:
    st.info("Every game on this day has started. Pick another day above to bet.", icon="⏱️")
for g in open_games.itertuples():
    render_game(g)
if not live_games.empty:
    with st.expander(f"🔴 In progress or already started ({len(live_games)}): betting closed"):
        for g in live_games.itertuples():
            render_game(g)
