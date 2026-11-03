"""Streamlit entry point: ``streamlit run src/analyst/dashboard/app.py``."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from analyst.agent import run_analysis, run_public_dataset
from analyst.contracts import AnalysisReport, Severity
from analyst.dashboard.charts import anomaly_chart, forecast_chart
from analyst.datasets import REGISTRY

_ICON = {Severity.INFO: "ℹ️", Severity.WARNING: "⚠️", Severity.ERROR: "⛔"}


@st.cache_data(show_spinner="Running analysis...")
def _analyze_public(name: str) -> AnalysisReport:
    return run_public_dataset(name)


@st.cache_data(show_spinner="Running analysis...")
def _analyze_upload(data: bytes, filename: str) -> AnalysisReport:
    return run_analysis(io.BytesIO(data), filename=filename)


def main() -> None:
    st.set_page_config(page_title="Data Analyst Agent", layout="wide")
    st.title("Autonomous Data Analyst Agent")

    with st.sidebar:
        source = st.radio("Data source", ["Public dataset", "Upload file"])
        if source == "Public dataset":
            name = st.selectbox("Dataset", sorted(REGISTRY))
            st.caption(REGISTRY[name].description)
            report = _analyze_public(name)
        else:
            upload = st.file_uploader("CSV or Parquet", type=["csv", "parquet"])
            if upload is None:
                st.info("Upload a file to start.")
                return
            report = _analyze_upload(upload.getvalue(), upload.name)

    for w in report.warnings:
        st.warning(w)

    overview, quality, anomalies, forecasts = st.tabs(["Insights", "Data quality", "Anomalies", "Forecast"])

    with overview:
        for insight in report.insights:
            st.markdown(f"{_ICON[insight.severity]} **{insight.title}**  \n{insight.detail}")

    with quality:
        st.metric("Quality score", f"{report.quality.score:.0f}/100")
        if report.quality.issues:
            st.dataframe(pd.DataFrame([i.model_dump(mode="json") for i in report.quality.issues]))
        st.subheader("Cleaning steps")
        st.dataframe(pd.DataFrame([s.model_dump() for s in report.cleaning.steps]))
        st.subheader("Column profile")
        st.dataframe(pd.DataFrame([c.model_dump() for c in report.profile.columns]))

    with anomalies:
        for result in report.anomalies:
            if result.metrics:
                cols = st.columns(len(result.metrics))
                for col, (k, v) in zip(cols, result.metrics.items()):
                    col.metric(k, f"{v:.2f}")
            st.plotly_chart(anomaly_chart(result), use_container_width=True)

    with forecasts:
        for result in report.forecasts:
            st.dataframe(pd.DataFrame({"model": result.metrics, "naive baseline": result.baseline_metrics}))
            st.plotly_chart(forecast_chart(result), use_container_width=True)


main()
