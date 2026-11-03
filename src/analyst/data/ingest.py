from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from analyst.contracts import DatasetSpec

SUPPORTED_SUFFIXES = (".csv", ".parquet")


def load_dataset(source: str | Path | IO[bytes], filename: str | None = None) -> pd.DataFrame:
    """Load a CSV or Parquet file into a raw frame.

    ``source`` may be a path or a binary file object (e.g. a Streamlit upload);
    for file objects the format is taken from ``filename`` or ``source.name``.
    """
    name = filename or (str(source) if isinstance(source, (str, Path)) else getattr(source, "name", ""))
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported file type {suffix!r}; expected one of {SUPPORTED_SUFFIXES}")


def infer_spec(raw: pd.DataFrame, name: str = "dataset") -> DatasetSpec:
    """Guess the timestamp column and numeric targets of a raw frame."""
    timestamp_col = _find_timestamp_col(raw)
    targets = [
        c for c in raw.columns
        if c != timestamp_col and pd.api.types.is_numeric_dtype(raw[c])
    ]
    if not targets:
        raise ValueError("no numeric target columns found")
    return DatasetSpec(name=name, timestamp_col=timestamp_col, target_cols=targets)


def _find_timestamp_col(raw: pd.DataFrame) -> str:
    for col in raw.columns:
        if pd.api.types.is_datetime64_any_dtype(raw[col]):
            return col
    for col in raw.columns:
        if pd.api.types.is_numeric_dtype(raw[col]):
            continue
        sample = raw[col].dropna().head(50)
        if len(sample) and pd.to_datetime(sample, errors="coerce").notna().all():
            return col
    raise ValueError("no timestamp column found")
