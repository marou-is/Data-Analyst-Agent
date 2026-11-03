"""Session state and cached loading shared by the dashboard pages."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Hashable

import pandas as pd
import streamlit as st

from analyst.agent import load_source, profile_source, run_analysis, run_public_dataset
from analyst.contracts import AnalysisReport, DatasetProfile
from analyst.datasets import SUPERSTORE

RAW = "table_raw"
FILENAME = "table_filename"
PROFILE = "table_profile"
SOURCE_KEY = "table_source_key"
"""Identifies what is loaded, so a lingering upload does not overwrite a newer choice."""

EXCEL_SUFFIXES = (".xlsx", ".xls")
UPLOAD_TYPES = ["csv", "xlsx", "xls", "parquet"]


@st.cache_data(show_spinner="Loading and profiling...", max_entries=8)
def load_table(data: bytes, filename: str, sheet_name: str | int = 0) -> tuple[pd.DataFrame, DatasetProfile]:
    """Raw frame and profile of a file given as bytes."""
    raw = load_source(io.BytesIO(data), filename=filename, sheet_name=sheet_name)
    return raw, profile_source(raw, filename=filename)


@st.cache_data(max_entries=8)
def excel_sheets(data: bytes) -> list[str]:
    return [str(s) for s in pd.ExcelFile(io.BytesIO(data)).sheet_names]


def is_excel(filename: str) -> bool:
    return Path(filename).suffix.lower() in EXCEL_SUFFIXES


def set_table(raw: pd.DataFrame, profile: DatasetProfile, filename: str, source_key: Hashable) -> None:
    st.session_state[RAW] = raw
    st.session_state[PROFILE] = profile
    st.session_state[FILENAME] = filename
    st.session_state[SOURCE_KEY] = source_key


def load_sample() -> None:
    """Load the Superstore sample into session state (button callback)."""
    path = SUPERSTORE.path
    raw, profile = load_table(path.read_bytes(), path.name)
    set_table(raw, profile, path.name, ("sample", SUPERSTORE.name))


def loaded_source_key() -> Hashable | None:
    return st.session_state.get(SOURCE_KEY)


def current_table() -> tuple[pd.DataFrame, DatasetProfile, str] | None:
    """The loaded raw frame, its profile and file name, if any."""
    if PROFILE not in st.session_state:
        return None
    return st.session_state[RAW], st.session_state[PROFILE], st.session_state[FILENAME]


@st.cache_data(show_spinner="Running analysis...", max_entries=4)
def analyze_public(name: str) -> AnalysisReport:
    return run_public_dataset(name)


@st.cache_data(show_spinner="Running analysis...", max_entries=4)
def analyze_upload(data: bytes, filename: str) -> AnalysisReport:
    return run_analysis(io.BytesIO(data), filename=filename)
