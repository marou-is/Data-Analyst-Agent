"""AI Analyst: ask a question in plain English and get an evidence-backed answer."""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

from analyst.agent.analyst import SUGGESTED_QUESTIONS, aggregated_metric, answer_question
from analyst.contracts import AnalystAnswer
from analyst.dashboard import state
from analyst.dashboard.charts import anomaly_chart, answer_chart, forecast_chart
from analyst.dashboard.components import eyebrow, md, page_header, panel, panel_heading, prompt_card
from analyst.dashboard.state import Workspace

INTENT_LABEL = {
    "breakdown": "Breakdown", "trend": "Trend", "anomalies": "Anomaly detection",
    "quality": "Data quality", "forecast": "Forecast", "overview": "Dataset overview",
}


def ask(question: str | None = None) -> None:
    """Answer ``question`` (or the text box) and store the result in the session."""
    if question is not None:
        st.session_state[state.QUESTION] = question
    text = (st.session_state.get(state.QUESTION) or "").strip()
    st.session_state[state.ANSWER_ERROR] = None
    if not text:
        st.session_state[state.ANSWER] = None
        return
    ws = state.workspace()
    try:
        st.session_state[state.ANSWER] = answer_question(text, ws.table, ws.profile, ws.report)
    except Exception as exc:  # noqa: BLE001 - surface analysis errors to the user
        st.session_state[state.ANSWER] = None
        st.session_state[state.ANSWER_ERROR] = f"I could not answer that on this dataset: {exc}"


def render(ws: Workspace) -> None:
    page_header("AI Analyst", "Ask your data.", f"Questions are answered from {ws.name}, with the evidence shown.")

    with prompt_card("question"):
        eyebrow("Your question")
        st.text_area("Question", key=state.QUESTION, placeholder="e.g. Which category generates the most revenue?",
                     label_visibility="collapsed", height=90)
        st.button("Analyse", icon=":material/auto_awesome:", type="primary", on_click=ask, key="ask_run")
        st.markdown('<div class="panel-note">Suggested questions</div>', unsafe_allow_html=True)
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, (col, question) in enumerate(zip(cols, SUGGESTED_QUESTIONS)):
            col.button(question, on_click=ask, args=(question,), key=f"suggest_{i}", width="stretch")

    if st.session_state.get(state.ANSWER_ERROR):
        st.error(st.session_state[state.ANSWER_ERROR])
    answer: AnalystAnswer | None = st.session_state.get(state.ANSWER)
    if answer is not None:
        _result(ws, answer)


def _result(ws: Workspace, answer: AnalystAnswer) -> None:
    with panel("answer"):
        st.markdown(f'<span class="chip">{escape(INTENT_LABEL[answer.intent])}</span>', unsafe_allow_html=True)
        panel_heading(answer.question)
        st.markdown(f"#### {md(answer.summary)}")
        fig = _chart(ws, answer)
        if fig is not None:
            st.plotly_chart(fig, key="answer_chart")
        if answer.evidence:
            eyebrow("Evidence")
            st.markdown("\n".join(f"- {md(line)}" for line in answer.evidence))
        if answer.table is not None and len(answer.table):
            with st.expander("Supporting data", icon=":material/table_view:"):
                st.dataframe(answer.table, hide_index=True, key="answer_table")
        st.caption(f"Answered by: {answer.engine}")


def _chart(ws: Workspace, answer: AnalystAnswer) -> go.Figure | None:
    metric = aggregated_metric(ws.report, answer.metric)
    if ws.report is not None and metric is not None:
        if answer.intent == "forecast":
            result = next((f for f in ws.report.forecasts if f.target_col == metric), None)
            if result is not None:
                return forecast_chart(result)
        if answer.intent == "anomalies":
            result = next((a for a in ws.report.anomalies if a.target_col == metric), None)
            if result is not None:
                return anomaly_chart(result)
    return answer_chart(answer)
