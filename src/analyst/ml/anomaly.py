from __future__ import annotations

import numpy as np
import pandas as pd

from analyst.contracts import AnalysisConfig, AnomalyResult, DatasetSpec
from analyst.ml.evaluation import classification_metrics

METHODS = ("seasonal_robust_zscore",)


def detect_anomalies(
    df: pd.DataFrame,
    spec: DatasetSpec,
    target_col: str,
    config: AnalysisConfig,
    labels: pd.Series | None = None,
) -> AnomalyResult:
    """Score every observation of ``target_col``; evaluate against ``labels`` if given."""
    if config.anomaly_method not in METHODS:
        raise ValueError(f"unknown anomaly method {config.anomaly_method!r}; available: {METHODS}")

    y = df[target_col]
    season = int(config.anomaly_params.get("season_length", spec.season_length or 1))
    score = _seasonal_robust_zscore(y, season)
    frame = pd.DataFrame(
        {"value": y, "score": score, "is_anomaly": score > config.anomaly_threshold},
        index=df.index,
    )

    metrics = {}
    if labels is not None:
        metrics = classification_metrics(labels.reindex(df.index, fill_value=False), frame["is_anomaly"])

    return AnomalyResult(
        method=config.anomaly_method,
        target_col=target_col,
        frame=frame,
        params={"season_length": season, "threshold": config.anomaly_threshold},
        metrics=metrics,
    )


def _seasonal_robust_zscore(y: pd.Series, season: int) -> pd.Series:
    resid = y - y.shift(season)
    centered = resid - resid.median()
    mad = 1.4826 * centered.abs().median()
    scale = mad if mad > 0 else (resid.std() or 1.0)
    return (centered.abs() / scale).fillna(0.0).astype(np.float64)
