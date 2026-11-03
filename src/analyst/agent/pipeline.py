from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from analyst.agent.insights import generate_insights
from analyst.contracts import (
    TIMESTAMP, AggregateResult, AnalysisConfig, AnalysisReport, CleaningStep, CleanResult, DatasetProfile,
    DatasetSpec, assert_clean_frame,
)
from analyst.data.aggregate import aggregate, default_aggregation
from analyst.data.clean import clean, clean_table
from analyst.data.ingest import infer_spec, load_dataset
from analyst.data.profile import profile, profile_dataset
from analyst.data.quality import check_quality
from analyst.datasets import Window, get_dataset, label_series
from analyst.ml.anomaly import detect_anomalies
from analyst.ml.forecast import forecast

TRANSACTIONAL_UNIQUE_RATIO = 0.9


def run_analysis(
    source: str | Path | IO[bytes] | pd.DataFrame,
    spec: DatasetSpec | None = None,
    config: AnalysisConfig | None = None,
    anomaly_windows: list[Window] | tuple[Window, ...] | None = None,
    filename: str | None = None,
) -> AnalysisReport:
    """Run the full analysis. Data-stage failures raise; ML-stage failures are
    recorded in ``report.warnings`` so the rest of the report is still usable."""
    config = config or AnalysisConfig()
    raw = source if isinstance(source, pd.DataFrame) else load_dataset(source, filename=filename)
    name = _source_name(source, filename)
    spec = spec or infer_spec(raw, name=name)
    table_profile = profile_dataset(raw, name=spec.name)

    aggregation = None
    if is_transactional(raw, spec):
        aggregation, spec, cleaning = _aggregate_transactions(raw, spec, table_profile, config)
        quality = table_profile.quality
    else:
        quality = check_quality(raw, spec)
        cleaning = clean(raw, spec)
    assert_clean_frame(cleaning.data, spec)
    data_profile = profile(cleaning.data, spec)

    updates = {}
    if spec.freq is None and data_profile.inferred_freq:
        updates["freq"] = data_profile.inferred_freq
    if spec.season_length is None:
        updates["season_length"] = infer_season_length(updates.get("freq", spec.freq))
    spec = spec.model_copy(update=updates)

    labels = label_series(cleaning.data.index, anomaly_windows) if anomaly_windows else None
    anomalies, forecasts, warnings = [], [], []
    for col in spec.target_cols:
        try:
            anomalies.append(detect_anomalies(cleaning.data, spec, col, config, labels=labels))
        except Exception as exc:  # noqa: BLE001 - keep the report usable
            warnings.append(f"anomaly detection failed for {col!r}: {exc}")
        try:
            forecasts.append(forecast(cleaning.data, spec, col, config))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"forecast failed for {col!r}: {exc}")

    report = AnalysisReport(
        spec=spec, config=config, quality=quality, cleaning=cleaning, profile=data_profile,
        anomalies=anomalies, forecasts=forecasts, warnings=warnings, table_profile=table_profile,
        aggregation=aggregation,
    )
    report.insights = generate_insights(report)
    return report


def is_transactional(raw: pd.DataFrame, spec: DatasetSpec) -> bool:
    """True when many rows share a timestamp (e.g. order lines), so the table must be
    aggregated per period rather than de-duplicated."""
    ts = raw[spec.timestamp_col].dropna()
    return len(ts) > 0 and ts.nunique() < TRANSACTIONAL_UNIQUE_RATIO * len(ts)


def _aggregate_transactions(
    raw: pd.DataFrame, spec: DatasetSpec, table_profile: DatasetProfile, config: AnalysisConfig
) -> tuple[AggregateResult, DatasetSpec, CleanResult]:
    table = clean_table(raw, table_profile)
    agg_spec = default_aggregation(table_profile, freq=config.aggregation_freq)
    agg_spec = agg_spec.model_copy(update={"timestamp_col": spec.timestamp_col})
    aggregation = aggregate(table.data, agg_spec)

    targets = [m.name for m in agg_spec.metrics if m.column in spec.target_cols] or aggregation.metric_names
    ts_spec = DatasetSpec(name=spec.name, timestamp_col=TIMESTAMP, target_cols=targets, freq=config.aggregation_freq)
    series = clean(aggregation.data.reset_index(), ts_spec)
    step = CleaningStep(
        name="aggregate", rows_affected=table.rows_after,
        description=f"Aggregated {table.rows_after:,} rows into {len(aggregation.data):,} '{agg_spec.freq}' periods by {spec.timestamp_col!r}",
    )
    cleaning = CleanResult(
        data=series.data, steps=[*table.steps, step, *series.steps],
        rows_before=len(raw), rows_after=len(series.data),
    )
    return aggregation, ts_spec, cleaning


def load_source(
    source: str | Path | IO[bytes], filename: str | None = None, sheet_name: str | int = 0
) -> pd.DataFrame:
    """Load a CSV/Excel/Parquet file as a raw frame (for display and charts)."""
    return load_dataset(source, filename=filename, sheet_name=sheet_name)


def profile_source(
    source: str | Path | IO[bytes] | pd.DataFrame,
    filename: str | None = None,
    sheet_name: str | int = 0,
) -> DatasetProfile:
    """Load any CSV/Excel/Parquet table (or DataFrame) and return its profile."""
    raw = source if isinstance(source, pd.DataFrame) else load_dataset(source, filename=filename, sheet_name=sheet_name)
    return profile_dataset(raw, name=_source_name(source, filename))


def _source_name(source, filename: str | None) -> str:
    if filename:
        return Path(filename).stem
    if isinstance(source, (str, Path)):
        return Path(source).stem
    return Path(str(getattr(source, "name", "dataset"))).stem


def run_public_dataset(name: str, config: AnalysisConfig | None = None) -> AnalysisReport:
    ds = get_dataset(name)
    return run_analysis(ds.path, spec=ds.spec, config=config, anomaly_windows=ds.anomaly_windows)


def infer_season_length(freq: str | None) -> int | None:
    """Daily cycle for sub-daily data, weekly for daily, yearly for weekly/monthly/quarterly."""
    if not freq:
        return None
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except ValueError:
        return None
    name = offset.name.upper()
    if name.startswith("W"):
        return 52 if offset.n == 1 else None
    if name.startswith("M") and offset.n == 1:
        return 12
    if name.startswith("Q") and offset.n == 1:
        return 4
    try:
        step = pd.Timedelta(offset.nanos, unit="ns")
    except (AttributeError, ValueError):
        return None
    day = pd.Timedelta("1D")
    if step < day and day % step == pd.Timedelta(0):
        return int(day / step)
    if step == day:
        return 7
    return None
