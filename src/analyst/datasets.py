"""Registry of public demo datasets and their ground-truth anomaly labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from analyst.config import RAW_DIR
from analyst.contracts import DatasetSpec

Window = tuple[str, str]


@dataclass(frozen=True)
class PublicDataset:
    name: str
    path: Path
    spec: DatasetSpec
    description: str
    source_url: str
    anomaly_windows: tuple[Window, ...] = field(default_factory=tuple)


NYC_TAXI = PublicDataset(
    name="nyc_taxi",
    path=RAW_DIR / "nyc_taxi.csv",
    spec=DatasetSpec(
        name="nyc_taxi",
        timestamp_col="timestamp",
        target_cols=["value"],
        freq="30min",
        season_length=48,
    ),
    description=(
        "Number of NYC taxi passengers aggregated in 30-minute buckets, "
        "2014-07-01 to 2015-01-31 (Numenta Anomaly Benchmark, realKnownCause)."
    ),
    source_url="https://github.com/numenta/NAB/blob/master/data/realKnownCause/nyc_taxi.csv",
    # labels/combined_windows.json: NYC marathon, Thanksgiving, Christmas,
    # New Year, January blizzard.
    anomaly_windows=(
        ("2014-10-30 15:30:00", "2014-11-03 22:30:00"),
        ("2014-11-25 12:00:00", "2014-11-29 19:00:00"),
        ("2014-12-23 11:30:00", "2014-12-27 18:30:00"),
        ("2014-12-29 21:30:00", "2015-01-03 04:30:00"),
        ("2015-01-24 20:30:00", "2015-01-29 03:30:00"),
    ),
)

REGISTRY: dict[str, PublicDataset] = {NYC_TAXI.name: NYC_TAXI}
"""Time-series datasets runnable through ``agent.run_public_dataset``."""


@dataclass(frozen=True)
class TabularDataset:
    name: str
    path: Path
    description: str
    source_url: str
    license: str


SUPERSTORE = TabularDataset(
    name="superstore",
    path=RAW_DIR / "superstore.csv",
    description=(
        "Tableau Sample Superstore: 9,994 US retail order lines (2014-2017) with order/ship "
        "dates, customer, segment, region, product category, sales, quantity, discount and profit."
    ),
    source_url="https://huggingface.co/datasets/An-j96/SuperstoreData",
    license="GPL-2.0",
)


def get_dataset(name: str) -> PublicDataset:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(REGISTRY)}") from None


def label_series(index: pd.DatetimeIndex, windows: tuple[Window, ...] | list[Window]) -> pd.Series:
    """Boolean series over ``index`` that is True inside any labelled window (inclusive)."""
    labels = pd.Series(False, index=index, name="is_anomaly")
    for start, end in windows:
        labels.loc[pd.Timestamp(start) : pd.Timestamp(end)] = True
    return labels
