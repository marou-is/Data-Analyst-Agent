"""Time-series analysis: insights, data quality, anomalies and forecasts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analyst.dashboard import state
from analyst.dashboard.charts import anomaly_chart, forecast_chart
from analyst.dashboard.tables import SEVERITY_ICON, quality_issues_frame
from analyst.datasets import REGISTRY

st.header("Time-series analysis")

with st.sidebar:
    source = st.segmented_control("Data source", ["Public dataset", "Upload file"], default="Public dataset", key="ts_source")
    if source == "Upload file":
        upload = st.file_uploader("CSV, Excel or Parquet", type=state.UPLOAD_TYPES, key="ts_upload")
        if upload is None:
            st.info("Upload a file to start.")
            st.stop()
        report = state.analyze_upload(upload.getvalue(), upload.name)
    else:
        name = st.selectbox("Dataset", sorted(REGISTRY), key="ts_dataset")
        st.caption(REGISTRY[name].description)
        report = state.analyze_public(name)

for w in report.warnings:
    st.warning(w)

overview, quality, anomalies, forecasts = st.tabs(["Insights", "Data quality", "Anomalies", "Forecast"])

with overview:
    for insight in report.insights:
        st.markdown(f"{SEVERITY_ICON[insight.severity]} **{insight.title}**  \n{insight.detail}")

with quality:
    st.metric("Quality score", f"{report.quality.score:.0f}/100")
    if report.quality.issues:
        st.dataframe(quality_issues_frame(report.quality), hide_index=True)
    st.subheader("Cleaning steps")
    st.dataframe(pd.DataFrame([s.model_dump() for s in report.cleaning.steps]), hide_index=True)
    st.subheader("Column profile")
    st.dataframe(pd.DataFrame([c.model_dump(mode="json") for c in report.profile.columns]), hide_index=True)

with anomalies:
    for result in report.anomalies:
        if result.metrics:
            cols = st.columns(len(result.metrics))
            for col, (k, v) in zip(cols, result.metrics.items()):
                col.metric(k, f"{v:.2f}")
        st.plotly_chart(anomaly_chart(result), width="stretch")

with forecasts:
    for result in report.forecasts:
        st.dataframe(pd.DataFrame({"model": result.metrics, "naive baseline": result.baseline_metrics}))
        st.plotly_chart(forecast_chart(result), width="stretch")
