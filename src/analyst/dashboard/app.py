"""Streamlit entry point: ``streamlit run src/analyst/dashboard/app.py``."""

from __future__ import annotations

import streamlit as st

from analyst.dashboard import state
from analyst.dashboard.components import footer, html_text
from analyst.dashboard.theme import CSS
from analyst.dashboard.views import ai_analyst, automations, explorer, overview

VIEWS = {
    "Overview": (overview, ":material/space_dashboard:"),
    "Data Explorer": (explorer, ":material/table_chart:"),
    "AI Analyst": (ai_analyst, ":material/auto_awesome:"),
    "Automations": (automations, ":material/bolt:"),
}

st.set_page_config(page_title="Northstar | Autonomous Data Analyst", page_icon="✦", layout="wide")
st.html(CSS)
state.init()

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">✦</div><div><div class="brand-name">northstar</div>'
        '<div class="brand-sub">Data analyst agent</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-caption">Workspace</div>', unsafe_allow_html=True)
    for page, (_, icon) in VIEWS.items():
        st.button(
            page, icon=icon, key=f"nav_{page.lower().replace(' ', '_')}", width="stretch",
            type="primary" if st.session_state[state.PAGE] == page else "secondary",
            on_click=state.go_to, args=(page,),
        )
    st.markdown('<div class="nav-caption">Active dataset</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="side-value">{html_text(st.session_state[state.DATASET_NAME])}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="nav-caption">System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-meta">Analyst engine: rule-based</div>'
        '<div class="side-meta">LLM: not connected</div>'
        '<div class="side-meta">Scheduler: not connected</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-caption">Northstar · MVP 0.1</div>', unsafe_allow_html=True)

try:
    workspace = state.workspace()
except Exception as exc:  # noqa: BLE001 - a broken file must not take the whole app down
    st.error(f"Could not load {st.session_state[state.DATASET_NAME]!r}: {exc}")
    st.button("Load the Superstore sample instead", on_click=state.load_sample)
    st.stop()

VIEWS[st.session_state[state.PAGE]][0].render(workspace)
footer()
