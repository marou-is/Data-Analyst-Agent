"""Data & profile: load a table and explore its profile and data quality."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from analyst.contracts import ColumnRole
from analyst.dashboard import state
from analyst.dashboard.charts import column_chart, missing_values_chart, role_breakdown_chart
from analyst.dashboard.tables import (
    column_overview_frame,
    quality_band,
    quality_issues_frame,
    sample_values,
)

st.header("Data & profile")

with st.container(border=True):
    left, right = st.columns([3, 1], vertical_alignment="center")
    upload = left.file_uploader("Upload a table (CSV, Excel or Parquet)", type=state.UPLOAD_TYPES, key="upload")
    right.button(
        "Load sample dataset (Superstore)", on_click=state.load_sample, icon=":material/dataset:",
        width="stretch", key="load_sample",
    )
    if upload is not None:
        data = upload.getvalue()
        sheet: str | int = 0
        if state.is_excel(upload.name):
            try:
                sheets = state.excel_sheets(data)
            except Exception as exc:  # noqa: BLE001 - show any reader error to the user
                st.error(f"Could not read workbook {upload.name!r}: {exc}")
                st.stop()
            if len(sheets) > 1:
                sheet = left.selectbox("Sheet", sheets, key="upload_sheet")
        source_key = ("upload", upload.file_id, sheet)
        if source_key != state.loaded_source_key():
            try:
                raw, profile = state.load_table(data, upload.name, sheet)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load {upload.name!r}: {exc}")
                st.stop()
            name = upload.name if sheet == 0 else f"{upload.name} [{sheet}]"
            state.set_table(raw, profile, name, source_key)

table = state.current_table()
if table is None:
    st.info("Upload a file or load the sample dataset to start.", icon=":material/upload_file:")
    st.stop()
raw, profile, filename = table

st.subheader(filename)
quality_score = profile.quality.score if profile.quality else None
card = {"border": True, "height": "stretch"}
metrics = st.columns([1, 1, 1, 1, 2, 1])
metrics[0].metric("Rows", f"{profile.n_rows:,}", **card)
metrics[1].metric("Columns", f"{profile.n_cols:,}", **card)
metrics[2].metric("Duplicate rows", f"{profile.n_duplicate_rows:,}", **card)
metrics[3].metric("Memory", f"{profile.memory_mb:,.1f} MB" if profile.memory_mb is not None else "n/a", **card)
metrics[4].metric(
    "Date range",
    f"{profile.start:%Y-%m-%d} to {profile.end:%Y-%m-%d}" if profile.start and profile.end else "n/a",
    **card,
)
if quality_score is None:
    metrics[5].metric("Quality score", "n/a", **card)
else:
    label, color = quality_band(quality_score)
    metrics[5].metric(
        "Quality score", f"{quality_score:.0f}/100", delta=label, delta_color=color, delta_arrow="off",
        help="Good from 85, fair from 60, poor below.", **card,
    )

quality_tab, columns_tab, explorer_tab, raw_tab = st.tabs(
    ["Data quality", "Columns", "Column explorer", "Raw data"]
)

with quality_tab:
    if profile.quality and profile.quality.issues:
        st.dataframe(
            quality_issues_frame(profile.quality), hide_index=True, key="quality_table",
            column_config={
                "severity": st.column_config.TextColumn(width="small"),
                "affected": st.column_config.NumberColumn(width="small", format="localized"),
                "message": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.success("No data-quality issues found.")
    if any(c.n_missing for c in profile.columns):
        st.plotly_chart(missing_values_chart(profile), width="stretch")
    else:
        st.info("No missing values in any column.", icon=":material/check_circle:")

with columns_tab:
    overview_col, roles_col = st.columns([3, 1])
    overview_col.dataframe(
        column_overview_frame(profile), hide_index=True, key="column_table",
        column_config={
            "% missing": st.column_config.NumberColumn(format="%.1f%%"),
            "key stats": st.column_config.TextColumn(width="large"),
        },
    )
    roles_col.plotly_chart(role_breakdown_chart(profile), width="stretch")

with explorer_tab:
    names = [c.name for c in profile.columns]
    numeric = profile.columns_with_role(ColumnRole.NUMERIC)
    name = st.selectbox(
        "Column", names, index=names.index(numeric[0]) if numeric else 0, key="explore_column", width=400,
    )
    col = profile.column(name)
    series = raw[name]
    facts = st.columns(4)
    facts[0].metric("Role", col.role.value if col.role else "n/a")
    facts[1].metric("Dtype", col.dtype)
    facts[2].metric("Missing", f"{col.n_missing:,} ({col.pct_missing:.1%})")
    facts[3].metric("Unique values", f"{col.n_unique:,}")
    fig = column_chart(col, series)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    elif col.role is ColumnRole.EMPTY:
        st.info("This column has no values.")
    else:
        st.caption("Sample values")
        st.dataframe(sample_values(series), hide_index=True)

with raw_tab:
    n_preview = min(100, profile.n_rows)
    st.caption(f"First {n_preview:,} of {profile.n_rows:,} rows")
    st.dataframe(raw.head(n_preview))
    st.download_button(
        "Download profile JSON", profile.model_dump_json(indent=2),
        file_name=f"{Path(filename).stem}_profile.json", mime="application/json", icon=":material/download:",
    )
