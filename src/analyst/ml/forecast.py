from __future__ import annotations

import numpy as np
import pandas as pd

from analyst.contracts import AnalysisConfig, DatasetSpec, ForecastResult
from analyst.ml.evaluation import forecast_metrics

MODELS = ("seasonal_naive",)


def forecast(
    df: pd.DataFrame, spec: DatasetSpec, target_col: str, config: AnalysisConfig
) -> ForecastResult:
    """Forecast ``config.forecast_horizon`` steps ahead, with a hold-out backtest
    of the same length compared against a last-value naive baseline."""
    if config.forecast_model not in MODELS:
        raise ValueError(f"unknown forecast model {config.forecast_model!r}; available: {MODELS}")

    y = df[target_col]
    h = config.forecast_horizon
    m = spec.season_length or 1
    if len(y) < h + 2 * m:
        raise ValueError(f"need at least {h + 2 * m} observations, got {len(y)}")

    train, test = y.iloc[:-h], y.iloc[-h:]
    backtest = pd.DataFrame({"y": test, "yhat": _seasonal_naive(train, h, m)}, index=test.index)
    baseline = np.full(h, train.iloc[-1])

    sigma = float((y - y.shift(m)).std())
    yhat = _seasonal_naive(y, h, m)
    future = pd.DataFrame(
        {"yhat": yhat, "yhat_lower": yhat - 1.96 * sigma, "yhat_upper": yhat + 1.96 * sigma},
        index=_future_index(y.index, h, spec.freq),
    )

    return ForecastResult(
        model=config.forecast_model,
        target_col=target_col,
        horizon=h,
        forecast=future,
        backtest=backtest,
        metrics=forecast_metrics(test, backtest["yhat"]),
        baseline_metrics=forecast_metrics(test, baseline),
        params={"season_length": m},
    )


def _seasonal_naive(y: pd.Series, h: int, m: int) -> np.ndarray:
    last_season = y.to_numpy(dtype=float)[-m:]
    return last_season[np.arange(h) % m]


def _future_index(index: pd.DatetimeIndex, h: int, freq: str | None) -> pd.DatetimeIndex:
    step = pd.tseries.frequencies.to_offset(freq) if freq else index.to_series().diff().median()
    return pd.DatetimeIndex([index[-1] + step * (i + 1) for i in range(h)], name=index.name)
