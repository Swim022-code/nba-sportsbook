"""Players: league leaders, player profiles and game logs from ESPN box scores."""

import plotly.graph_objects as go
import streamlit as st

from src import players, style
from src.charts import BLUE, GRAY, ORANGE, styled
from src.ui import STAT_SHORT, get_games, house_edge_on, player_games, season_label, sidebar

games = get_games()
sidebar()
pg = player_games()

style.hero("Players", "Season averages, league leaders and game logs for every NBA player", "⛹️")
if pg.empty:
    st.warning("No player data yet. Click **🔄 Refresh data** in the sidebar.")
    st.stop()

seasons = sorted(pg["season"].unique(), reverse=True)
f1, f2, f3 = st.columns([1, 1, 2])
season = f1.selectbox("Season", seasons, format_func=season_label)
min_gp = f2.number_input("Min. games played", 1, 82, 5)
avg = players.season_averages(pg, season, min_games=int(min_gp))
if avg.empty:
    st.info("No games for that season yet.")
    st.stop()
team_f = f3.multiselect("Teams", sorted(avg["team"].unique()), placeholder="All teams")
view = avg[avg["team"].isin(team_f)] if team_f else avg

tab_leaders, tab_player = st.tabs(["🏅 League leaders", "👤 Player profile"])

with tab_leaders:
    cards = []
    for stat, label in [("ppg", "Points"), ("rpg", "Rebounds"), ("apg", "Assists"), ("tpm", "3-pointers")]:
        top = view.nlargest(1, stat).iloc[0]
        cards.append({"label": f"{label} leader", "value": f"{top[stat]:.1f}", "name": top["name"], "team": top["team"],
                      "headshot": top["headshot"] if isinstance(top["headshot"], str) else None})
    style.leader_cards(cards)
    st.write("")
    sort_by = st.radio("Sort by", ["ppg", "rpg", "apg", "tpm", "mpg", "fg_pct"], horizontal=True,
                       format_func={"ppg": "Points", "rpg": "Rebounds", "apg": "Assists", "tpm": "3PM",
                                    "mpg": "Minutes", "fg_pct": "FG%"}.get)
    table = view.sort_values(sort_by, ascending=False).head(50).copy()
    table.insert(0, "rank", range(1, len(table) + 1))
    st.dataframe(
        table[["rank", "headshot", "name", "team", "position", "gp", "mpg", "ppg", "rpg", "apg", "tpm", "spg", "bpg",
               "fg_pct", "fg3_pct", "ft_pct"]],
        hide_index=True, width="stretch", row_height=42, height=560,
        column_config={
            "rank": "#", "headshot": st.column_config.ImageColumn(" ", width="small"), "name": "Player",
            "team": "Team", "position": "Pos", "gp": "GP",
            **{k: st.column_config.NumberColumn(k.upper(), format="%.1f")
               for k in ["mpg", "ppg", "rpg", "apg", "tpm", "spg", "bpg"]},
            "fg_pct": st.column_config.NumberColumn("FG%", format="percent"),
            "fg3_pct": st.column_config.NumberColumn("3P%", format="percent"),
            "ft_pct": st.column_config.NumberColumn("FT%", format="percent"),
        },
    )
    st.caption(f"Source: ESPN box scores, {view['gp'].sum():,} player-games. Averages count games played only.")

with tab_player:
    names = dict(zip(view["player_id"], view["name"] + " (" + view["team"] + ")", strict=True))
    pid = st.selectbox("Find a player", list(names), format_func=names.get,
                       index=0, placeholder="Type a name...")
    me = avg.set_index("player_id").loc[pid]
    style.player_card(me["name"], me["team"], me["position"], me["headshot"], {
        "GP": f"{me['gp']:.0f}", "MIN": f"{me['mpg']:.1f}", "PTS": f"{me['ppg']:.1f}", "REB": f"{me['rpg']:.1f}",
        "AST": f"{me['apg']:.1f}", "3PM": f"{me['tpm']:.1f}",
        "FG%": f"{me['fg_pct']:.1%}" if me["fg_pct"] == me["fg_pct"] else "-"})
    st.write("")
    log = players.played(pg[(pg["player_id"] == pid) & (pg["season"] == season)]).sort_values("game_date")
    stat = st.radio("Chart stat", list(STAT_SHORT), horizontal=True, format_func=STAT_SHORT.get, key="log_stat")
    fig = go.Figure()
    fig.add_bar(x=log["game_date"], y=log[stat], name=STAT_SHORT[stat], marker_color=BLUE, opacity=0.8,
                customdata=log["opponent"], hovertemplate="%{x|%b %d} vs %{customdata}: %{y}<extra></extra>")
    fig.add_scatter(x=log["game_date"], y=log[stat].rolling(5, min_periods=1).mean(), name="5-game average",
                    line=dict(color=ORANGE, width=2), mode="lines")
    fig.add_hline(y=log[stat].mean(), line_dash="dot", line_color=GRAY,
                  annotation_text=f"season avg {log[stat].mean():.1f}", annotation_position="top left")
    st.plotly_chart(styled(fig, f"{me['name']}: {STAT_SHORT[stat]} by game"), width="stretch")

    # props for this player's next game
    nxt = games[~games["completed"] & ((games["home_team"] == me["team"]) | (games["away_team"] == me["team"]))]
    nxt = nxt.sort_values("tipoff_utc").head(1)
    if not nxt.empty:
        g = nxt.iloc[0]
        opp = g["away_team"] if g["home_team"] == me["team"] else g["home_team"]
        menu = [m for m in players.prop_menu(pg, me["team"], opp, house_edge_on(), n_players=8)
                if m["player_id"] == pid]
        if menu:
            st.markdown(f"**Props for the next game** ({g['away_team']} @ {g['home_team']}, "
                        f"{g['tipoff_utc'].tz_convert('America/New_York'):%a %b %d}): "
                        + " · ".join(f"{STAT_SHORT[m['stat']]} {m['line']:g}" for m in menu)
                        + ". Bet them on the **🏀 Sportsbook**.")

    games_tbl = log.sort_values("game_date", ascending=False)
    st.dataframe(
        games_tbl[["game_date", "opponent", "is_home", "starter", "minutes", "pts", "reb", "ast", "fg3m", "fgm", "fga",
                   "ftm", "fta", "stl", "blk", "tov", "plus_minus"]],
        hide_index=True, width="stretch",
        column_config={"game_date": st.column_config.DateColumn("Date", format="MMM D"), "opponent": "Opp",
                       "is_home": st.column_config.CheckboxColumn("Home"),
                       "starter": st.column_config.CheckboxColumn("Started"),
                       "minutes": "MIN", "pts": "PTS", "reb": "REB", "ast": "AST", "fg3m": "3PM", "fgm": "FGM",
                       "fga": "FGA", "ftm": "FTM", "fta": "FTA", "stl": "STL", "blk": "BLK", "tov": "TO",
                       "plus_minus": "+/-"},
    )
