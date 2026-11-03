"""Streamlit entry point: ``streamlit run src/analyst/dashboard/app.py``."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).parent / "app_pages"

st.set_page_config(page_title="Data Analyst Agent", layout="wide")

page = st.navigation(
    [
        st.Page(PAGES_DIR / "profile.py", title="Data & profile", icon=":material/table_chart:", default=True),
        st.Page(PAGES_DIR / "timeseries.py", title="Time-series analysis", icon=":material/show_chart:"),
    ],
    position="top",
)

st.title("Autonomous Data Analyst Agent")
page.run()
