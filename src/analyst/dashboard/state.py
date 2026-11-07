"""Session state, navigation and cached analysis shared by the dashboard views."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import streamlit as st

from analyst.agent import load_source, profile_source, run_analysis
from analyst.agent.analyst import prepare_table
from analyst.contracts import AnalysisReport, DatasetProfile
from analyst.datasets import SUPERSTORE

PAGES = ("Overview", "Data Explorer", "AI Analyst", "Automations")
PAGE = "page"
SOURCE = "source"
"""``(file bytes, file name, sheet)`` of the active dataset."""
DATASET_NAME = "dataset_name"
QUESTION = "question"
ANSWER = "answer"
ANSWER_ERROR = "answer_error"
AUTOMATIONS = "automations"

EXCEL_SUFFIXES = (".xlsx", ".xls")
UPLOAD_TYPES = ["csv", "xlsx", "xls", "parquet"]
SAMPLE_NAME = "Superstore sample"


@dataclass(frozen=True)
class Workspace:
    name: str
    raw: pd.DataFrame
    profile: DatasetProfile
    table: pd.DataFrame
    """Clean table used for business answers (see ``agent.analyst.prepare_table``)."""
    report: AnalysisReport | None
    report_error: str | None


@st.cache_data(show_spinner="Loading and profiling...", max_entries=8)
def load_table(data: bytes, filename: str, sheet_name: str | int = 0) -> tuple[pd.DataFrame, DatasetProfile]:
    """Raw frame and profile of a file given as bytes."""
    raw = load_source(io.BytesIO(data), filename=filename, sheet_name=sheet_name)
    return raw, profile_source(raw, filename=filename)


@st.cache_data(show_spinner="Analysing your data...", max_entries=4)
def analyze(data: bytes, filename: str, sheet_name: str | int = 0) -> tuple[pd.DataFrame, AnalysisReport | None, str | None]:
    """Clean table plus the full analysis report, or the reason the analysis could not run."""
    raw, profile = load_table(data, filename, sheet_name)
    table = prepare_table(raw, profile)
    try:
        return table, run_analysis(raw, filename=filename), None
    except Exception as exc:  # noqa: BLE001 - profile-only datasets are still useful
        return table, None, str(exc)


@st.cache_data(max_entries=8)
def excel_sheets(data: bytes) -> list[str]:
    return [str(s) for s in pd.ExcelFile(io.BytesIO(data)).sheet_names]


def is_excel(filename: str) -> bool:
    return Path(filename).suffix.lower() in EXCEL_SUFFIXES


def init() -> None:
    """Default page and the Superstore sample on first visit."""
    st.session_state.setdefault(PAGE, PAGES[0])
    st.session_state.setdefault(AUTOMATIONS, [])
    st.session_state.setdefault(ANSWER, None)
    if SOURCE not in st.session_state:
        load_sample()


def set_source(data: bytes, filename: str, sheet: str | int, name: str) -> None:
    st.session_state[SOURCE] = (data, filename, sheet)
    st.session_state[DATASET_NAME] = name
    st.session_state[ANSWER] = None


def load_sample() -> None:
    path = SUPERSTORE.path
    set_source(path.read_bytes(), path.name, 0, SAMPLE_NAME)


def workspace() -> Workspace:
    data, filename, sheet = st.session_state[SOURCE]
    raw, profile = load_table(data, filename, sheet)
    table, report, error = analyze(data, filename, sheet)
    return Workspace(st.session_state[DATASET_NAME], raw, profile, table, report, error)


def go_to(page: str) -> None:
    st.session_state[PAGE] = page
