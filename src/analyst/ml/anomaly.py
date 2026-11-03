from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, STL

from analyst.contracts import AnalysisConfig, AnomalyResult, DatasetSpec
from analyst.ml.evaluation import classification_metrics

MIN_WEEKLY_CYCLES = 8

Scorer = Callable[[pd.Series, int, DatasetSpec, dict[str, Any]], tuple[pd.Series, dict[str, Any]]]


def detect_anomalies(
    df: pd.DataFrame,
    spec: DatasetSpec,
    target_col: str,
    config: AnalysisConfig,
    labels: pd.Series | None = None,
) -> AnomalyResult:
    """Score every observation of ``target_col``; evaluate against ``labels`` if given.

    ``config.anomaly_params`` may set ``season_length`` (overrides the spec),
    ``window`` (centred rolling mean of the score, default 1 = no smoothing) and
    method-specific options (``seasonal``/``robust`` for STL, ``periods`` for MSTL).
    """
    if config.anomaly_method not in METHODS:
        raise ValueError(f"unknown anomaly method {config.anomaly_method!r}; available: {METHODS}")

    y = df[target_col]
    params = dict(config.anomaly_params)
    season = int(params.pop("season_length", spec.season_length or 1))
    window = int(params.pop("window", 1))
    if window < 1:
        raise ValueError(f"anomaly_params['window'] must be >= 1, got {window}")

    score, used = _SCORERS[config.anomaly_method](y, season, spec, params)
    if window > 1:
        score = score.rolling(window, center=True, min_periods=1).mean()
    frame = pd.DataFrame(
        {"value": y, "score": score.astype(np.float64), "is_anomaly": score > config.anomaly_threshold},
        index=df.index,
    )

    metrics = {}
    if labels is not None:
        metrics = classification_metrics(labels.reindex(df.index, fill_value=False), frame["is_anomaly"])

    return AnomalyResult(
        method=config.anomaly_method,
        target_col=target_col,
        frame=frame,
        params={"season_length": season, "threshold": config.anomaly_threshold, **used,
                **({"window": window} if window > 1 else {})},
        metrics=metrics,
    )


def robust_zscore(resid: pd.Series) -> pd.Series:
    """Absolute MAD z-score; falls back to the std when the MAD is zero."""
    centered = resid - resid.median()
    mad = 1.4826 * centered.abs().median()
    scale = mad if mad > 0 else (resid.std() or 1.0)
    return (centered.abs() / scale).fillna(0.0).astype(np.float64)


def _seasonal_robust_zscore(
    y: pd.Series, season: int, spec: DatasetSpec, params: dict[str, Any]
) -> tuple[pd.Series, dict[str, Any]]:
    return robust_zscore(y - y.shift(season)), {}


def _stl_robust_zscore(
    y: pd.Series, season: int, spec: DatasetSpec, params: dict[str, Any]
) -> tuple[pd.Series, dict[str, Any]]:
    _require_seasons("stl_robust_zscore", y, [season])
    seasonal = int(params.get("seasonal", 7))
    robust = bool(params.get("robust", True))
    fit = STL(_filled(y), period=season, seasonal=seasonal, robust=robust, **_loess_jumps(season, seasonal)).fit()
    resid = pd.Series(np.asarray(fit.resid), index=y.index)
    return robust_zscore(resid), {"seasonal": seasonal, "robust": robust}


def _mstl_robust_zscore(
    y: pd.Series, season: int, spec: DatasetSpec, params: dict[str, Any]
) -> tuple[pd.Series, dict[str, Any]]:
    periods = sorted({int(p) for p in params.get("periods") or _default_periods(y, season, spec)})
    _require_seasons("mstl_robust_zscore", y, periods)
    robust = bool(params.get("robust", True))
    stl_kwargs = {"robust": robust, **_loess_jumps(max(periods), 7)}
    fit = MSTL(_filled(y), periods=periods, stl_kwargs=stl_kwargs).fit()
    resid = pd.Series(np.asarray(fit.resid), index=y.index)
    return robust_zscore(resid), {"periods": periods, "robust": robust}


def _loess_jumps(period: int, seasonal: int) -> dict[str, int]:
    """R ``stl`` defaults: evaluate each LOESS every ~10% of its window and interpolate.

    Results are practically identical to statsmodels' exact fit (jump 1) but 10-40x faster
    on long periods such as a weekly cycle of half-hourly data."""
    trend = math.ceil(1.5 * period / (1 - 1.5 / seasonal))
    trend += trend % 2 == 0
    low_pass = period + 1 + period % 2
    return {
        "seasonal_jump": math.ceil(seasonal / 10),
        "trend_jump": math.ceil(trend / 10),
        "low_pass_jump": math.ceil(low_pass / 10),
    }


def _default_periods(y: pd.Series, season: int, spec: DatasetSpec) -> list[int]:
    """The spec's season, plus a weekly cycle when the season is one day of sub-daily data
    and the series covers at least ``MIN_WEEKLY_CYCLES`` weeks (with fewer, the weekly
    component overfits and residuals get noisier)."""
    periods = [season]
    step = _step(y.index, spec.freq)
    if step is not None and step * season == pd.Timedelta("1D") and len(y) >= MIN_WEEKLY_CYCLES * 7 * season:
        periods.append(7 * season)
    return periods


def _step(index: pd.DatetimeIndex, freq: str | None) -> pd.Timedelta | None:
    try:
        if freq:
            return pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    except (ValueError, TypeError):
        pass
    if len(index) < 2:
        return None
    return pd.Timedelta(index.to_series().diff().median())


def _require_seasons(method: str, y: pd.Series, periods: list[int]) -> None:
    if min(periods) < 2:
        raise ValueError(f"{method} needs a season_length >= 2 (set it in the spec or anomaly_params)")
    need = 2 * max(periods)
    if y.notna().sum() < need:
        raise ValueError(
            f"{method} needs at least two full seasons ({need} observations for period "
            f"{max(periods)}), got {int(y.notna().sum())}"
        )


def _filled(y: pd.Series) -> pd.Series:
    return y.interpolate(limit_direction="both").astype(np.float64)


_SCORERS: dict[str, Scorer] = {
    "seasonal_robust_zscore": _seasonal_robust_zscore,
    "stl_robust_zscore": _stl_robust_zscore,
    "mstl_robust_zscore": _mstl_robust_zscore,
}
METHODS = tuple(_SCORERS)
