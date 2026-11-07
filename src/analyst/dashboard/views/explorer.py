"""Data Explorer: upload a table, preview it and inspect its profile and quality."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from analyst.contracts import ColumnRole
from analyst.dashboard import state
from analyst.dashboard.charts import column_chart, missing_values_chart, role_breakdown_chart
from analyst.dashboard.components import page_header, panel, panel_heading
from analyst.dashboard.state import Workspace
from analyst.dashboard.tables import column_overview_frame, quality_band, quality_issues_frame, sample_values

SEEN_UPLOAD = "seen_upload"
"""Last upload made active, so a file left in the uploader does not replace a later sample choice."""


def handle_upload() -> None:
    """Make the uploaded file the active dataset (once per file and sheet)."""
    upload = st.file_uploader("Upload a CSV, Excel or Parquet file", type=state.UPLOAD_TYPES, key="upload")
    if upload is None:
        return
    data, sheet = upload.getvalue(), 0
    if state.is_excel(upload.name):
        try:
            sheets = state.excel_sheets(data)
        except Exception as exc:  # noqa: BLE001 - show any reader error to the user
            st.error(f"Could not read workbook {upload.name!r}: {exc}")
            return
        if len(sheets) > 1:
            sheet = st.selectbox("Sheet", sheets, key="upload_sheet")
    source_key = ("upload", upload.file_id, sheet)
    if source_key == st.session_state.get(SEEN_UPLOAD):
        return
    st.session_state[SEEN_UPLOAD] = source_key
    try:
        state.load_table(data, upload.name, sheet)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load {upload.name!r}: {exc}")
        return
    state.set_source(data, upload.name, sheet, upload.name if sheet == 0 else f"{upload.name} [{sheet}]")
    st.rerun()


def render(ws: Workspace) -> None:
    page_header("Data Explorer", "Know your data.", "Upload a table, then check its structure and quality before analysis.")
    profile = ws.profile

    left, right = st.columns([1.4, 1], gap="medium")
    with left, panel("upload"):
        panel_heading("Data source", "The agent profiles and analyses whatever you load here.")
        handle_upload()
        st.button("Use the Superstore sample", on_click=state.load_sample, icon=":material/dataset:", key="load_sample")
    with right, panel("snapshot"):
        panel_heading(ws.name, "Dataset snapshot")
        cards = st.columns(2)
        cards[0].metric("Rows", f"{profile.n_rows:,}")
        cards[1].metric("Columns", f"{profile.n_cols:,}")
        cards = st.columns(2)
        cards[0].metric("Duplicate rows", f"{profile.n_duplicate_rows:,}")
        if profile.quality is None:
            cards[1].metric("Quality score", "n/a")
        else:
            label, color = quality_band(profile.quality.score)
            cards[1].metric("Quality score", f"{profile.quality.score:.0f}/100", label, delta_color=color,
                            delta_arrow="off", help="Good from 85, fair from 60, poor below.")
        if profile.start and profile.end:
            st.caption(f"Dates from {profile.start:%Y-%m-%d} to {profile.end:%Y-%m-%d}")

    with panel("details"):
        preview_tab, columns_tab, quality_tab, explorer_tab = st.tabs(
            ["Data preview", "Column profile", "Missing values & quality", "Column explorer"]
        )
        with preview_tab:
            n_preview = min(100, profile.n_rows)
            st.caption(f"First {n_preview:,} of {profile.n_rows:,} rows")
            st.dataframe(ws.raw.head(n_preview), key="preview_table")
            st.download_button(
                "Download profile JSON", profile.model_dump_json(indent=2),
                file_name=f"{Path(ws.name).stem}_profile.json", mime="application/json", icon=":material/download:",
            )
        with columns_tab:
            overview_col, roles_col = st.columns([3, 1])
            overview_col.dataframe(
                column_overview_frame(profile), hide_index=True, key="column_table",
                column_config={
                    "% missing": st.column_config.NumberColumn(format="%.1f%%"),
                    "key stats": st.column_config.TextColumn(width="large"),
                },
            )
            roles_col.plotly_chart(role_breakdown_chart(profile), key="roles_chart")
        with quality_tab:
            _quality(ws)
        with explorer_tab:
            _column_explorer(ws)


def _quality(ws: Workspace) -> None:
    profile = ws.profile
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
        st.plotly_chart(missing_values_chart(profile), key="missing_chart")
    else:
        st.info("No missing values in any column.", icon=":material/check_circle:")


def _column_explorer(ws: Workspace) -> None:
    profile = ws.profile
    names = [c.name for c in profile.columns]
    numeric = profile.columns_with_role(ColumnRole.NUMERIC)
    name = st.selectbox("Column", names, index=names.index(numeric[0]) if numeric else 0, key="explore_column",
                        width=400)
    col = profile.column(name)
    series = ws.raw[name]
    facts = st.columns(4)
    facts[0].metric("Role", col.role.value if col.role else "n/a")
    facts[1].metric("Dtype", col.dtype)
    facts[2].metric("Missing", f"{col.n_missing:,} ({col.pct_missing:.1%})")
    facts[3].metric("Unique values", f"{col.n_unique:,}")
    fig = column_chart(col, series)
    if fig is not None:
        st.plotly_chart(fig, key="column_chart")
    elif col.role is ColumnRole.EMPTY:
        st.info("This column has no values.")
    else:
        st.caption("Sample values")
        st.dataframe(sample_values(series), hide_index=True)
