from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from analyst.contracts import ColumnRole, DatasetSpec

SUPPORTED_SUFFIXES = (".csv", ".xlsx", ".xls", ".parquet")
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def load_dataset(
    source: str | Path | IO[bytes], filename: str | None = None, sheet_name: str | int = 0
) -> pd.DataFrame:
    """Load a CSV, Excel or Parquet file into a raw frame.

    ``source`` may be a path or a binary file object (e.g. a Streamlit upload);
    for file objects the format is taken from ``filename`` or ``source.name``.
    CSVs are decoded with the first encoding in ``CSV_ENCODINGS`` that works.
    """
    name = filename or (str(source) if isinstance(source, (str, Path)) else getattr(source, "name", ""))
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return _read_csv(source)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(source, sheet_name=sheet_name)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported file type {suffix!r}; expected one of {SUPPORTED_SUFFIXES}")


def _read_csv(source: str | Path | IO[bytes]) -> pd.DataFrame:
    for encoding in CSV_ENCODINGS:
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not decode CSV with any of {CSV_ENCODINGS}")


def infer_spec(raw: pd.DataFrame, name: str = "dataset") -> DatasetSpec:
    """Use the first datetime column as timestamp and numeric-role columns as targets
    (identifiers such as ``Row ID`` or ``Postal Code`` are excluded)."""
    from analyst.data.profile import infer_role

    roles = {col: infer_role(raw[col])[0] for col in raw.columns}
    timestamps = [c for c, r in roles.items() if r is ColumnRole.DATETIME]
    if not timestamps:
        raise ValueError("no timestamp column found")
    targets = [c for c, r in roles.items() if r is ColumnRole.NUMERIC]
    if not targets:
        raise ValueError("no numeric target columns found")
    return DatasetSpec(name=name, timestamp_col=timestamps[0], target_cols=targets)
