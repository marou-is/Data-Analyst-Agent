from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from analyst.agent.insights import generate_insights
from analyst.contracts import AnalysisConfig, AnalysisReport, DatasetSpec, assert_clean_frame
from analyst.data.clean import clean
from analyst.data.ingest import infer_spec, load_dataset
from analyst.data.profile import profile
from analyst.data.quality import check_quality
from analyst.datasets import Window, get_dataset, label_series
from analyst.ml.anomaly import detect_anomalies
from analyst.ml.forecast import forecast


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
    spec = spec or infer_spec(raw, name=Path(filename or str(getattr(source, "name", "dataset"))).stem)

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
        anomalies=anomalies, forecasts=forecasts, warnings=warnings,
    )
    report.insights = generate_insights(report)
    return report


def run_public_dataset(name: str, config: AnalysisConfig | None = None) -> AnalysisReport:
    ds = get_dataset(name)
    return run_analysis(ds.path, spec=ds.spec, config=config, anomaly_windows=ds.anomaly_windows)


def infer_season_length(freq: str | None) -> int | None:
    """Daily cycle for sub-daily data, weekly cycle for daily data, else unknown."""
    if not freq:
        return None
    try:
        step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    except (ValueError, TypeError):
        return None
    day = pd.Timedelta("1D")
    if step < day and day % step == pd.Timedelta(0):
        return int(day / step)
    if step == day:
        return 7
    return None
