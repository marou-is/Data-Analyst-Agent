"""Overview: headline KPIs, revenue trend, category mix and the agent's findings."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from analyst.agent.analyst import aggregated_metric, find_dimension, find_measure, format_value
from analyst.contracts import ColumnRole, Severity
from analyst.dashboard import state
from analyst.dashboard.charts import breakdown_donut_chart, trend_area_chart
from analyst.dashboard.components import SEVERITY_COLOR, finding, page_header, panel, panel_heading, prompt_card
from analyst.dashboard.state import Workspace
from analyst.dashboard.tables import quality_band

MAX_FINDINGS = 4
MAX_HIGHLIGHTS = 6


def greeting(hour: int) -> str:
    if hour < 12:
        return "Good morning"
    return "Good afternoon" if hour < 18 else "Good evening"


def render(ws: Workspace) -> None:
    page_header(
        "Overview", f"{greeting(datetime.now().hour)}.",
        f"Here's what's happening in {ws.name}: {len(ws.raw):,} records, {len(ws.raw.columns)} columns.",
    )
    revenue = find_measure("revenue", ws.profile)
    profit = find_measure("profit", ws.profile)
    profit = profit if profit != revenue else None
    date_col = next(iter(ws.profile.columns_with_role(ColumnRole.DATETIME)), None)

    _kpis(ws, revenue, profit)
    left, right = st.columns([1.65, 1], gap="medium")
    with left, panel("trend"):
        _trend(ws, revenue, date_col)
    with right, panel("mix"):
        _mix(ws, revenue)
    left, right = st.columns([1.65, 1], gap="medium")
    with left, panel("findings"):
        _findings(ws)
    with right, prompt_card("ask"):
        st.markdown('<div class="eyebrow">AI Analyst</div>', unsafe_allow_html=True)
        st.markdown("### Ask your data anything.")
        st.markdown(
            '<div class="panel-note">Plain-English questions, answers backed by the numbers in your table.</div>',
            unsafe_allow_html=True,
        )
        st.button("Ask a question", icon=":material/auto_awesome:", type="primary",
                  on_click=state.go_to, args=("AI Analyst",), key="overview_ask")


def _kpis(ws: Workspace, revenue: str | None, profit: str | None) -> None:
    cards = st.columns(4, gap="medium")
    table = ws.table
    if revenue is not None:
        cards[0].metric(f"Total {revenue.lower()}", format_value(revenue, float(table[revenue].sum())))
    if profit is not None:
        total = float(table[profit].sum())
        margin = f"{total / float(table[revenue].sum()):.1%} margin" if revenue and table[revenue].sum() else None
        cards[1].metric(f"Total {profit.lower()}", format_value(profit, total), margin, delta_color="off", delta_arrow="off")
    else:
        cards[1].metric("Columns", len(ws.raw.columns))
    duplicates = len(ws.raw) - len(table)
    cards[2].metric("Records", f"{len(ws.raw):,}",
                    f"{duplicates:,} duplicate{'s' if duplicates > 1 else ''} removed" if duplicates else "no duplicates", delta_color="off", delta_arrow="off")
    if ws.profile.quality is None:
        cards[3].metric("Data quality", "n/a")
        return
    score = ws.profile.quality.score
    label, color = quality_band(score)
    cards[3].metric("Data quality", f"{score:.0f}/100", label, delta_color=color, delta_arrow="off")


def _trend(ws: Workspace, revenue: str | None, date_col: str | None) -> None:
    if revenue is None or date_col is None:
        panel_heading("Trend", "A trend needs a date column and a numeric column.")
        return
    dates = pd.to_datetime(ws.table[date_col], errors="coerce")
    series = ws.table[revenue].groupby(dates).sum().resample("W").sum().rename(revenue)
    panel_heading(f"{revenue} trend", f"Weekly {revenue.lower()} by {date_col}; amber points are weeks with unusual days.")
    st.plotly_chart(trend_area_chart(series, _highlights(ws, revenue, series)), key="overview_trend")


def _highlights(ws: Workspace, revenue: str, weekly: pd.Series) -> pd.Series | None:
    metric = aggregated_metric(ws.report, revenue)
    result = next((a for a in (ws.report.anomalies if ws.report else []) if a.target_col == metric), None)
    if result is None:
        return None
    flagged = result.frame[result.frame["is_anomaly"]]
    if flagged.empty:
        return None
    weeks = flagged["score"].resample("W").max().dropna().nlargest(MAX_HIGHLIGHTS).index
    return weekly.reindex(weeks).dropna().sort_index()


def _mix(ws: Workspace, revenue: str | None) -> None:
    dim = find_dimension("", ws.profile)
    if revenue is None or dim is None:
        panel_heading("Mix", "A breakdown needs a categorical column and a numeric column.")
        return
    values = ws.table.groupby(dim, observed=True)[revenue].sum()
    panel_heading(f"{revenue} by {dim.lower()}", f"Share of total {revenue.lower()}.")
    st.plotly_chart(breakdown_donut_chart(values), key="overview_mix")


def _findings(ws: Workspace) -> None:
    panel_heading("AI findings", "What the agent noticed while analysing this dataset.")
    if ws.report is None:
        st.info(f"The full analysis could not run on this dataset: {ws.report_error}")
        return
    rank = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    insights = sorted(ws.report.insights, key=lambda i: rank[i.severity])[:MAX_FINDINGS]
    if not insights:
        st.caption("No notable findings.")
    for insight in insights:
        finding(insight.title, insight.detail, SEVERITY_COLOR[insight.severity])
