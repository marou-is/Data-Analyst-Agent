from __future__ import annotations

import pandas as pd

from analyst.contracts import TIMESTAMP, CleaningStep, CleanResult, DatasetSpec


def clean(raw: pd.DataFrame, spec: DatasetSpec) -> CleanResult:
    """Turn a raw frame into a clean frame (see ``analyst.contracts``)."""
    steps: list[CleaningStep] = []
    df = raw[[spec.timestamp_col, *spec.target_cols]].copy()
    rows_before = len(df)

    df[spec.timestamp_col] = pd.to_datetime(df[spec.timestamp_col], errors="coerce")
    bad_ts = int(df[spec.timestamp_col].isna().sum())
    if bad_ts:
        df = df.dropna(subset=[spec.timestamp_col])
        steps.append(CleaningStep(name="drop_bad_timestamps", description="Dropped rows with unparseable timestamps", rows_affected=bad_ts))

    for col in spec.target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df = df.sort_values(spec.timestamp_col)
    dupes = int(df[spec.timestamp_col].duplicated().sum())
    if dupes:
        df = df.drop_duplicates(subset=[spec.timestamp_col], keep="first")
        steps.append(CleaningStep(name="drop_duplicate_timestamps", description="Kept first row per duplicated timestamp", rows_affected=dupes))

    df = df.set_index(spec.timestamp_col).rename_axis(TIMESTAMP)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    n_missing = int(df[spec.target_cols].isna().sum().sum())
    if n_missing:
        df[spec.target_cols] = df[spec.target_cols].interpolate(method="time", limit_direction="both")
        steps.append(CleaningStep(name="interpolate_missing", description="Time-interpolated missing target values", rows_affected=n_missing))

    return CleanResult(data=df, steps=steps, rows_before=rows_before, rows_after=len(df))
