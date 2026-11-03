from __future__ import annotations

import pandas as pd

from analyst.contracts import ColumnProfile, DatasetProfile, DatasetSpec


def profile(df: pd.DataFrame, spec: DatasetSpec) -> DatasetProfile:
    """Summary statistics of a clean frame."""
    columns = []
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s) and s.notna().any()
        columns.append(
            ColumnProfile(
                name=col,
                dtype=str(s.dtype),
                n_missing=int(s.isna().sum()),
                pct_missing=float(s.isna().mean()) if len(s) else 0.0,
                n_unique=int(s.nunique()),
                mean=float(s.mean()) if numeric else None,
                std=float(s.std()) if numeric and len(s) > 1 else None,
                min=float(s.min()) if numeric else None,
                median=float(s.median()) if numeric else None,
                max=float(s.max()) if numeric else None,
            )
        )
    return DatasetProfile(
        n_rows=len(df),
        n_cols=df.shape[1],
        start=df.index.min().to_pydatetime() if len(df) else None,
        end=df.index.max().to_pydatetime() if len(df) else None,
        inferred_freq=infer_freq(df.index),
        columns=columns,
    )


def infer_freq(index: pd.DatetimeIndex) -> str | None:
    if len(index) < 3:
        return None
    freq = pd.infer_freq(index)
    if freq is None:
        freq = pd.tseries.frequencies.to_offset(index.to_series().diff().median()).freqstr
    return freq
