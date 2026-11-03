"""Deterministic synthetic time series with known defects, for offline tests and demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyst.contracts import DatasetSpec
from analyst.datasets import Window


def make_synthetic_series(
    n_days: int = 28, freq: str = "30min", seed: int = 0
) -> tuple[pd.DataFrame, DatasetSpec, list[Window]]:
    """Return ``(raw_frame, spec, anomaly_windows)``.

    The raw frame mimics a loaded CSV: string timestamps, a ``value`` column with
    daily and weekly seasonality plus trend, three injected anomaly windows,
    ~1% missing values and a few duplicated rows appended at the end.
    """
    rng = np.random.default_rng(seed)
    per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    index = pd.date_range("2024-01-01", periods=n_days * per_day, freq=freq)
    t = np.arange(len(index))

    value = (
        100.0
        + 0.01 * t
        + 15.0 * np.sin(2 * np.pi * t / per_day)
        + 6.0 * np.sin(2 * np.pi * t / (7 * per_day))
        + rng.normal(0, 2.0, len(index))
    )

    windows: list[Window] = []
    for day, length, delta in ((9, 3, 60.0), (16, per_day // 2, -45.0), (23, 2, 80.0)):
        start = day * per_day + per_day // 3
        value[start : start + length] += delta
        windows.append((str(index[start]), str(index[start + length - 1])))

    missing = rng.choice(len(index), size=max(1, len(index) // 100), replace=False)
    value[missing] = np.nan

    raw = pd.DataFrame({"timestamp": index.strftime("%Y-%m-%d %H:%M:%S"), "value": value})
    raw = pd.concat([raw, raw.iloc[[5, 50, 500]]], ignore_index=True)

    spec = DatasetSpec(
        name="synthetic", timestamp_col="timestamp", target_cols=["value"], freq=freq,
        season_length=per_day,
    )
    return raw, spec, windows
