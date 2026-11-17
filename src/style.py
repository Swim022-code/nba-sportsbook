"""Visual styling: one stylesheet plus small HTML components (page banner, game
cards, odds 'buttons', win-probability bar, house-edge badge).

Streamlit widgets (forms, buttons, charts) stay native; these helpers only render
display-only HTML, so the app keeps working even if the CSS fails to load.
"""

from __future__ import annotations

from html import escape

import streamlit as st

NAVY, NAVY_2, RED = "#0b2545", "#13315c", "#c8102e"
GREEN_TEXT, MUTED, BORDER, SURFACE = "#0a7d3b", "#5f6670", "#e3e6eb", "#f4f6f9"


def logo_url(abbr: str) -> str:
    """ESPN team logo (all 30 NBA teams follow this pattern)."""
    return f"https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{abbr.lower()}.png"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, p, li, label, h1, h2, h3, h4, input, textarea, button, td, th,
.stMarkdown, [data-testid="stMetricValue"], [data-testid="stWidgetLabel"] {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
/* keep Streamlit's icon font for its built-in icons (expander arrows, sidebar toggle) */
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {{
  font-family: 'Material Symbols Rounded' !important;
}}
.block-container {{ padding-top: 2.2rem; max-width: 1250px; }}
[data-testid="stAppDeployButton"] {{ display: none; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}

/* page banner */
.sb-hero {{
  background: linear-gradient(120deg, {NAVY} 0%, {NAVY_2} 65%, #1d4e89 100%);
  color: #fff; border-radius: 18px; padding: 22px 28px; margin: 0 0 18px 0;
  display: flex; align-items: center; gap: 18px; box-shadow: 0 6px 18px rgba(11,37,69,.18);
}}
.sb-hero .icon {{ font-size: 44px; line-height: 1; }}
.sb-hero h1 {{ color: #fff; margin: 0; padding: 0; font-size: 30px; font-weight: 800; }}
.sb-hero p {{ color: #c9d6ea; margin: 4px 0 0 0; font-size: 15px; }}
.sb-hero .badge {{ margin-left: auto; flex-shrink: 0; }}

/* KPI cards: style native st.metric */
[data-testid="stMetric"] {{
  background: #fff; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 14px 18px; box-shadow: 0 1px 3px rgba(16,24,40,.05);
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-weight: 600; font-size: 14px; }}
[data-testid="stMetricValue"] {{ font-weight: 800; color: {NAVY}; }}
[data-testid="stSidebar"] [data-testid="stMetric"] {{ background: {SURFACE}; }}

/* game card */
.sb-card {{ background: #fff; border: 1px solid {BORDER}; border-radius: 16px;
  padding: 16px 18px 12px 18px; margin-bottom: 6px; box-shadow: 0 1px 4px rgba(16,24,40,.06); }}
.sb-card-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
.sb-time {{ font-size: 13px; font-weight: 700; color: {NAVY}; background: {SURFACE};
  padding: 4px 10px; border-radius: 999px; }}
.sb-live {{ font-size: 13px; font-weight: 700; color: {RED}; background: #fdecee;
  padding: 4px 10px; border-radius: 999px; }}
.sb-grid {{ display: grid; grid-template-columns: minmax(170px, 2.2fr) repeat(3, minmax(84px, 1fr));
  gap: 8px; align-items: center; }}
.sb-head {{ font-size: 12px; font-weight: 700; color: {MUTED}; text-transform: uppercase;
  letter-spacing: .05em; text-align: center; }}
.sb-team {{ display: flex; align-items: center; gap: 10px; min-width: 0; }}
.sb-team img {{ width: 34px; height: 34px; object-fit: contain; }}
.sb-team .name {{ font-weight: 700; color: #111827; font-size: 15px; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }}
.sb-team .rec {{ color: {MUTED}; font-size: 12px; }}
.sb-odd {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; text-align: center;
  padding: 7px 4px; line-height: 1.25; }}
.sb-odd .line {{ font-weight: 700; color: #111827; font-size: 14px; }}
.sb-odd .price {{ font-weight: 700; font-size: 13px; color: {NAVY_2}; }}
.sb-odd .price.plus {{ color: {GREEN_TEXT}; }}

/* win probability bar */
.sb-prob {{ margin-top: 12px; }}
.sb-prob .labels {{ display: flex; justify-content: space-between; font-size: 12px; color: {MUTED};
  font-weight: 600; margin-bottom: 4px; }}
.sb-prob .bar {{ display: flex; height: 8px; border-radius: 999px; overflow: hidden; background: {BORDER}; gap: 2px; }}
.sb-prob .bar div {{ height: 100%; }}
.sb-foot {{ font-size: 12px; color: {MUTED}; margin-top: 8px; }}

/* badges */
.sb-badge {{ display: inline-block; font-size: 12px; font-weight: 700; padding: 5px 11px; border-radius: 999px;
  white-space: nowrap; }}

/* bet slip summary */
.sb-slip {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 6px 0 12px 0; }}
.sb-slip > div {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 8px 10px; }}
.sb-slip .k {{ font-size: 11px; font-weight: 700; color: {MUTED}; text-transform: uppercase; letter-spacing: .04em; }}
.sb-slip .v {{ font-size: 18px; font-weight: 800; color: {NAVY}; }}
.sb-slip .s {{ font-size: 12px; color: {MUTED}; }}
.sb-slip .neg {{ color: {RED}; }} .sb-slip .pos {{ color: {GREEN_TEXT}; }}
.sb-badge.on {{ background: #fff4e5; color: #8a4b00; border: 1px solid #f5c98b; }}
.sb-badge.off {{ background: #e8f6ee; color: {GREEN_TEXT}; border: 1px solid #a8dcbc; }}
.sb-hero .sb-badge.on {{ background: rgba(255,244,229,.95); }}

/* sidebar brand */
.sb-brand {{ display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 18px; color: {NAVY}; }}
.sb-brand span {{ font-size: 26px; }}

/* house panel */
.sb-house {{ background: linear-gradient(120deg, #fff7ed, #fff); border: 1px solid #f5d0a4;
  border-radius: 16px; padding: 16px 20px; }}
.sb-house h4 {{ margin: 0 0 4px 0; color: #7a3e00; }}
.sb-house p {{ margin: 0; color: #5b4a3a; font-size: 14px; }}

/* player card */
.sb-player {{ display: flex; align-items: center; gap: 18px; background: #fff; border: 1px solid {BORDER};
  border-radius: 16px; padding: 14px 18px; box-shadow: 0 1px 4px rgba(16,24,40,.06); }}
.sb-player img.head {{ width: 92px; height: 68px; object-fit: cover; border-radius: 10px; background: {SURFACE}; }}
.sb-player img.logo {{ width: 34px; height: 34px; }}
.sb-player .nm {{ font-size: 22px; font-weight: 800; color: {NAVY}; line-height: 1.1; }}
.sb-player .sub {{ color: {MUTED}; font-size: 14px; }}
.sb-player .stats {{ display: flex; gap: 22px; margin-left: auto; flex-wrap: wrap; }}
.sb-player .stats div {{ text-align: center; }}
.sb-player .stats b {{ display: block; font-size: 22px; color: #111827; }}
.sb-player .stats span {{ font-size: 12px; color: {MUTED}; font-weight: 700; }}

/* league-leader cards */
.sb-leaders {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
.sb-leader {{ display: flex; align-items: center; gap: 12px; background: #fff; border: 1px solid {BORDER};
  border-radius: 14px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(16,24,40,.05); min-width: 0; }}
.sb-leader img {{ width: 56px; height: 42px; object-fit: cover; border-radius: 8px; background: {SURFACE}; }}
.sb-leader .k {{ font-size: 12px; font-weight: 700; color: {MUTED}; text-transform: uppercase; letter-spacing: .04em; }}
.sb-leader .v {{ font-size: 22px; font-weight: 800; color: {NAVY}; line-height: 1.1; }}
.sb-leader .n {{ font-size: 13px; font-weight: 600; color: #111827; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }}
@media (max-width: 900px) {{ .sb-leaders {{ grid-template-columns: 1fr 1fr; }} }}

/* podium */
.sb-podium {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; align-items: end; }}
.sb-podium .p {{ border-radius: 16px; padding: 14px; text-align: center; color: #111827;
  border: 1px solid {BORDER}; background: #fff; box-shadow: 0 1px 4px rgba(16,24,40,.06); }}
.sb-podium .p .medal {{ font-size: 30px; }}
.sb-podium .p .u {{ font-weight: 800; font-size: 17px; }}
.sb-podium .p .n {{ font-size: 13px; color: {MUTED}; }}
.sb-podium .first {{ padding-top: 26px; background: linear-gradient(180deg, #fff8e1, #fff); border-color: #f2d27a; }}

/* house ledgers */
.sb-ledgers {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.sb-ledger {{ border-radius: 16px; padding: 14px 18px; border: 1px solid {BORDER}; background: #fff; }}
.sb-ledger.edge {{ background: linear-gradient(120deg, #fff7ed, #fff); border-color: #f5d0a4; }}
.sb-ledger.fair {{ background: linear-gradient(120deg, #ecfdf3, #fff); border-color: #a8dcbc; }}
.sb-ledger h4 {{ margin: 0 0 6px 0; }}
.sb-ledger .big {{ font-size: 26px; font-weight: 800; }}
.sb-ledger .row {{ display: flex; justify-content: space-between; font-size: 13px; color: {MUTED}; }}
.sb-ledger.active {{ box-shadow: 0 0 0 3px rgba(19,49,92,.25); }}
@media (max-width: 700px) {{ .sb-ledgers, .sb-podium {{ grid-template-columns: 1fr; }} }}

.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
@media (max-width: 700px) {{
  .sb-grid {{ grid-template-columns: 1fr 1fr 1fr; }}
  .sb-grid .sb-team, .sb-grid .sb-headteam {{ grid-column: 1 / -1; }}
  .sb-hero {{ padding: 16px; }} .sb-hero h1 {{ font-size: 24px; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def edge_badge(house_edge: bool) -> str:
    if house_edge:
        return '<span class="sb-badge on">🏦 House edge ON · ~4.5% margin</span>'
    return '<span class="sb-badge off">⚖️ Fair odds · 0% margin</span>'


def hero(title: str, subtitle: str, icon: str, badge_html: str = "") -> None:
    st.markdown(
        f'<div class="sb-hero"><div class="icon">{icon}</div><div><h1>{escape(title)}</h1>'
        f'<p>{escape(subtitle)}</p></div><div class="badge">{badge_html}</div></div>',
        unsafe_allow_html=True,
    )


def _price(odds: int) -> str:
    txt = f"+{odds}" if odds > 0 else str(odds)
    return f'<div class="price{" plus" if odds > 0 else ""}">{txt}</div>'


def _line(x: float) -> str:
    return f"+{x:g}" if x > 0 else f"{x:g}"


def _team(abbr: str, name: str, record: str | None) -> str:
    return (f'<div class="sb-team"><img src="{logo_url(abbr)}" alt="{escape(abbr)}"/>'
            f'<div style="min-width:0"><div class="name">{escape(name)}</div>'
            f'<div class="rec">{escape(record or "")}</div></div></div>')


def game_card(g, lines: dict, status_html: str, foot: str) -> None:
    """Sportsbook-style card: teams with logos, then spread / total / moneyline 'buttons'."""
    away, home = g.away_team, g.home_team
    p_home = lines["p_home"]
    fav_color, dog_color = NAVY_2, "#9fb3cc"
    away_col, home_col = (dog_color, fav_color) if p_home >= 0.5 else (fav_color, dog_color)
    html = f"""
<div class="sb-card">
  <div class="sb-card-top">{status_html}</div>
  <div class="sb-grid">
    <div class="sb-head" style="text-align:left">Matchup</div>
    <div class="sb-head">Spread</div><div class="sb-head">Total</div><div class="sb-head">Moneyline</div>
    {_team(away, g.away_team_name, g.away_record)}
    <div class="sb-odd"><div class="line">{_line(lines['away_spread'])}</div>{_price(lines['spread_odds'])}</div>
    <div class="sb-odd"><div class="line">O {lines['total']:g}</div>{_price(lines['total_odds'])}</div>
    <div class="sb-odd"><div class="line">&nbsp;</div>{_price(lines['away_ml'])}</div>
    {_team(home, g.home_team_name, g.home_record)}
    <div class="sb-odd"><div class="line">{_line(lines['home_spread'])}</div>{_price(lines['spread_odds'])}</div>
    <div class="sb-odd"><div class="line">U {lines['total']:g}</div>{_price(lines['total_odds'])}</div>
    <div class="sb-odd"><div class="line">&nbsp;</div>{_price(lines['home_ml'])}</div>
  </div>
  <div class="sb-prob">
    <div class="labels"><span>{escape(away)} {1 - p_home:.0%}</span><span>Model win probability</span>
      <span>{escape(home)} {p_home:.0%}</span></div>
    <div class="bar"><div style="width:{(1 - p_home) * 100:.1f}%;background:{away_col}"></div>
      <div style="width:{p_home * 100:.1f}%;background:{home_col}"></div></div>
  </div>
  <div class="sb-foot">{foot}</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def slip_summary(payout: float, profit: float, p_model: float, p_implied: float, ev: float) -> None:
    """Three compact stat boxes for the bet slip."""
    ev_cls = "pos" if ev > 0.005 else "neg" if ev < -0.005 else ""
    st.markdown(
        f'<div class="sb-slip">'
        f'<div><div class="k">Pays</div><div class="v">{payout:,.2f}</div>'
        f'<div class="s">+{profit:,.2f} profit</div></div>'
        f'<div><div class="k">Model chance</div><div class="v">{p_model:.0%}</div>'
        f'<div class="s">odds imply {p_implied:.0%}</div></div>'
        f'<div><div class="k">Expected value</div><div class="v {ev_cls}">{ev:+.2f}</div>'
        f'<div class="s">avg. profit per bet</div></div></div>',
        unsafe_allow_html=True,
    )


def player_card(name: str, team: str, position: str | None, headshot: str | None, stats: dict[str, str]) -> None:
    head = f'<img class="head" src="{escape(headshot)}" alt=""/>' if isinstance(headshot, str) and headshot else ""
    cells = "".join(f"<div><b>{escape(v)}</b><span>{escape(k)}</span></div>" for k, v in stats.items())
    st.markdown(
        f'<div class="sb-player">{head}<img class="logo" src="{logo_url(team)}" alt="{escape(team)}"/>'
        f'<div><div class="nm">{escape(name)}</div><div class="sub">{escape(team)} · {escape(position or "")}</div></div>'
        f'<div class="stats">{cells}</div></div>',
        unsafe_allow_html=True,
    )


def leader_cards(cards: list[dict]) -> None:
    """cards: [{label, value, name, team, headshot}]"""
    html = "".join(
        f'<div class="sb-leader"><img src="{escape(c["headshot"] or logo_url(c["team"]))}" alt=""/>'
        f'<div style="min-width:0"><div class="k">{escape(c["label"])}</div><div class="v">{escape(c["value"])}</div>'
        f'<div class="n">{escape(c["name"])} · {escape(c["team"])}</div></div></div>' for c in cards)
    st.markdown(f'<div class="sb-leaders">{html}</div>', unsafe_allow_html=True)


def podium(rows: list[dict], currency: str) -> None:
    """Top-3 leaderboard cards (rows sorted best first)."""
    medals = ["🥇", "🥈", "🥉"]
    order = [1, 0, 2] if len(rows) >= 3 else list(range(len(rows)))  # 2nd, 1st, 3rd like a podium
    cards = []
    for i in order:
        r = rows[i]
        cls = "p first" if i == 0 else "p"
        cards.append(f'<div class="{cls}"><div class="medal">{medals[i]}</div><div class="u">{escape(r["username"])}</div>'
                     f'<div class="n">{r["balance"]:,.2f} {currency} · net {r["net"]:+,.2f}</div></div>')
    st.markdown(f'<div class="sb-podium">{"".join(cards)}</div>', unsafe_allow_html=True)


def house_ledgers(edge: dict, fair: dict, active_book: str, currency: str) -> None:
    """Side-by-side ledgers for the house-edge book and the fair book."""
    def card(s: dict, cls: str, title: str) -> str:
        color = GREEN_TEXT if s["house_profit"] >= 0 else RED
        return (f'<div class="sb-ledger {cls}{" active" if s["book"] == active_book else ""}">'
                f'<h4>{title}</h4><div class="big" style="color:{color}">{s["house_profit"]:+,.2f}</div>'
                f'<div class="row"><span>House profit ({currency})</span><span>hold {s["hold_pct"]:.1%}</span></div>'
                f'<div class="row"><span>Taken in (settled)</span><span>{s["handle"]:,.2f}</span></div>'
                f'<div class="row"><span>Paid back to bettors</span><span>{s["paid_out"]:,.2f}</span></div>'
                f'<div class="row"><span>Settled / open wagers</span><span>{s["settled_bets"]} / {s["open_bets"]}</span></div>'
                f'<div class="row"><span>Riding on open wagers</span><span>{s["open_stakes"]:,.2f}</span></div></div>')
    st.markdown(f'<div class="sb-ledgers">{card(edge, "edge", "🏦 House edge ON book")}'
                f'{card(fair, "fair", "⚖️ Fair-odds book")}</div>', unsafe_allow_html=True)
