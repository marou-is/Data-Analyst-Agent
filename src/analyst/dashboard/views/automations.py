"""Automations: define recurring analyses. Scheduling is not connected yet; "Run now" works."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import streamlit as st

from analyst.agent.analyst import answer_question
from analyst.dashboard import state
from analyst.dashboard.components import eyebrow, md, page_header, panel, panel_heading
from analyst.dashboard.state import Workspace

WORKFLOWS = {
    "Full analysis report": "Give me an overview of the dataset",
    "Anomaly check": "Are there unusual values in my dataset?",
    "Data quality check": "What are the main data quality issues?",
    "Revenue trend": "How has revenue changed over time?",
}
CADENCES = ("Daily", "Weekly", "Monthly")
OUTPUTS = ("In-app report", "Email (not connected)", "Slack (not connected)")


@dataclass
class Automation:
    name: str
    workflow: str
    cadence: str
    output: str
    dataset: str
    created: datetime = field(default_factory=datetime.now)
    last_run: datetime | None = None
    last_result: str | None = None


def automations() -> list[Automation]:
    return st.session_state.setdefault(state.AUTOMATIONS, [])


def run(index: int) -> None:
    item = automations()[index]
    ws = state.workspace()
    try:
        item.last_result = answer_question(WORKFLOWS[item.workflow], ws.table, ws.profile, ws.report).summary
    except Exception as exc:  # noqa: BLE001 - a failed run is recorded, not raised
        item.last_result = f"Run failed: {exc}"
    item.last_run = datetime.now()


def render(ws: Workspace) -> None:
    page_header("Automations", "Put analysis on autopilot.",
                "Save recurring checks on your data. Run them on demand today; scheduling comes next.")
    items = automations()
    left, right = st.columns([1.4, 1], gap="medium")
    with left, panel("create"):
        panel_heading("New automation", f"Runs against the active dataset ({ws.name}).")
        with st.form("new_automation", clear_on_submit=True, border=False):
            name = st.text_input("Name", placeholder="e.g. Monday revenue check", key="automation_name")
            workflow = st.selectbox("Workflow", list(WORKFLOWS), key="automation_workflow")
            cols = st.columns(2)
            cadence = cols[0].selectbox("Cadence", CADENCES, index=1, key="automation_cadence")
            output = cols[1].selectbox("Deliver to", OUTPUTS, key="automation_output")
            if st.form_submit_button("Create automation", icon=":material/add:", type="primary"):
                items.append(Automation(name.strip() or workflow, workflow, cadence, output, ws.name))
                st.toast(f"Automation created: {items[-1].name}")
    with right, panel("health"):
        panel_heading("Automation health")
        cols = st.columns(2)
        cols[0].metric("Active", len(items))
        cols[1].metric("Runs today", sum(i.last_run is not None and i.last_run.date() == datetime.now().date()
                                         for i in items))
        st.caption("Scheduler not connected: automations run only when you press Run now.")

    with panel("list"):
        panel_heading("Your automations")
        if not items:
            st.caption("No automations yet. Create one to see it here.")
        for i, item in enumerate(items):
            text, button = st.columns([5, 1], vertical_alignment="center")
            with text:
                st.markdown(f"**{md(item.name)}**")
                st.caption(f"{item.workflow} · {item.cadence} · {item.output} · {item.dataset}")
                if item.last_run is not None:
                    eyebrow(f"Last run {item.last_run:%Y-%m-%d %H:%M}")
                    st.markdown(md(item.last_result or ""))
            button.button("Run now", key=f"automation_run_{i}", on_click=run, args=(i,), icon=":material/play_arrow:")
            if i < len(items) - 1:
                st.divider()
