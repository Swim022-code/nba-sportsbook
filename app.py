"""NBA Mock Sportsbook - Streamlit entry point.

Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="NBA Mock Sportsbook", page_icon="🏀", layout="wide")

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/sportsbook.py", title="Sportsbook", icon="🏀", url_path="sportsbook"),
    st.Page("views/my_bets.py", title="My Bets", icon="🎟️", url_path="my_bets"),
    st.Page("views/data_analysis.py", title="Data Analysis", icon="📊", url_path="data_analysis"),
    st.Page("views/ai_analyst.py", title="AI Analyst", icon="🤖", url_path="ai_analyst"),
]
st.navigation(pages).run()
