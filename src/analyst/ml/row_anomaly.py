"""Row-level (tabular) anomaly detection with human-readable reasons."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from analyst.contracts import AnalysisConfig, ColumnRole, DatasetProfile, RowAnomalyResult

METHODS = ("isolation_forest", "robust_zscore")
MAX_SAMPLES = 10_000
"""Default trees are grown on (up to) the whole table: sklearn's 256-row subsamples cap
the depth at 8, so heavy natural tails (e.g. Superstore Sales) score as high as truly
implausible rows."""


def detect_row_anomalies(table: pd.DataFrame, profile: DatasetProfile, config: AnalysisConfig) -> RowAnomalyResult:
    """Score every row of ``table`` using its NUMERIC-role columns (per ``profile``).

    Both methods flag the ``ceil(contamination * n_rows)`` highest scores, except
    ``robust_zscore`` with an explicit ``threshold`` param. ``config.row_anomaly_params``:

    * ``ratios``: list of ``[numerator, denominator]`` pairs added as derived features,
      e.g. ``[["Profit", "Sales"]]`` (division by zero becomes missing).
    * ``one_hot`` (isolation_forest only): ``True`` for every CATEGORICAL column with at
      most ``max_categories`` (default 10) distinct values, or an explicit list of columns.
    * ``n_estimators`` (default 200), ``max_samples`` (default ``min(n_rows, MAX_SAMPLES)``),
      ``random_state`` (default 0): isolation forest.
    * ``transform`` (isolation_forest only): ``"asinh"`` (default) compresses the heavy
      tails of the robust z-scores before fitting; ``"none"`` uses them as is.
    * ``threshold``: robust_zscore cut-off on the max per-feature MAD z-score.

    Missing feature values are imputed with the column median. Tables without usable
    numeric features get an all-normal result (``params["skipped"]`` says why).
    """
    method = config.row_anomaly_method
    if method not in METHODS:
        raise ValueError(f"unknown row anomaly method {method!r}; available: {METHODS}")
    params = dict(config.row_anomaly_params)
    contamination = config.row_anomaly_contamination

    raw = _numeric_features(table, profile, params.get("ratios") or [])
    used: dict[str, Any] = {"contamination": contamination}
    if params.get("ratios"):
        used["ratios"] = [list(pair) for pair in params["ratios"]]
    if raw.shape[1] == 0 or len(table) == 0:
        frame = pd.DataFrame(
            {"score": np.zeros(len(table)), "is_anomaly": np.zeros(len(table), dtype=bool), "reason": ""},
            index=table.index,
        )
        skipped = "empty table" if len(table) == 0 else "no numeric features"
        return RowAnomalyResult(method=method, frame=frame, params={**used, "skipped": skipped})

    filled = raw.fillna(raw.median())
    z = (filled - filled.median()) / _robust_scale(filled)

    if method == "isolation_forest":
        transform = params.get("transform", "asinh")
        if transform not in ("asinh", "none"):
            raise ValueError(f"unknown transform {transform!r}; available: ('asinh', 'none')")
        one_hot = _one_hot_columns(table, profile, params)
        X = np.arcsinh(z) if transform == "asinh" else z
        if one_hot:
            X = pd.concat([X, _dummies(table, one_hot)], axis=1)
        n_estimators = int(params.get("n_estimators", 200))
        random_state = int(params.get("random_state", 0))
        max_samples = int(params.get("max_samples", min(len(table), MAX_SAMPLES)))
        model = IsolationForest(
            n_estimators=n_estimators, max_samples=min(max_samples, len(table)),
            contamination=contamination, random_state=random_state,
        ).fit(X.to_numpy(dtype=np.float64))
        score = pd.Series(-model.score_samples(X.to_numpy(dtype=np.float64)), index=table.index)
        used.update(n_estimators=n_estimators, max_samples=max_samples, random_state=random_state,
                    transform=transform, one_hot=one_hot)
        is_anomaly = _top_fraction(score, contamination)
    else:
        score = z.abs().max(axis=1)
        if "threshold" in params:
            used["threshold"] = float(params["threshold"])
            is_anomaly = score > used["threshold"]
        else:
            is_anomaly = _top_fraction(score, contamination)

    flagged = is_anomaly.to_numpy(dtype=bool)
    reason = np.full(len(table), "", dtype=object)
    for pos in np.flatnonzero(flagged):
        reason[pos] = _reason(raw.iloc[pos], z.iloc[pos])
    frame = pd.DataFrame(
        {"score": score.to_numpy(dtype=np.float64), "is_anomaly": flagged, "reason": pd.array(reason, dtype="str")},
        index=table.index,
    )
    return RowAnomalyResult(
        method=method,
        frame=frame,
        features=list(raw.columns),
        params=used,
        metrics={"n_anomalies": float(flagged.sum()), "pct_anomalies": float(flagged.mean())},
    )


def _numeric_features(table: pd.DataFrame, profile: DatasetProfile, ratios: list) -> pd.DataFrame:
    cols = [c for c in profile.columns_with_role(ColumnRole.NUMERIC) if c in table.columns]
    feats = pd.DataFrame({c: pd.to_numeric(table[c], errors="coerce").astype(np.float64) for c in cols}, index=table.index)
    for pair in ratios:
        if len(pair) != 2:
            raise ValueError(f"ratio must be a [numerator, denominator] pair, got {pair!r}")
        num, den = pair
        missing = [c for c in (num, den) if c not in table.columns]
        if missing:
            raise ValueError(f"ratio columns not in table: {missing}")
        a = pd.to_numeric(table[num], errors="coerce").astype(np.float64)
        b = pd.to_numeric(table[den], errors="coerce").astype(np.float64)
        feats[f"{num}/{den}"] = (a / b.where(b != 0)).replace([np.inf, -np.inf], np.nan)
    return feats.loc[:, feats.notna().any()]


def _robust_scale(df: pd.DataFrame) -> pd.Series:
    """Per-column MAD scale, falling back to IQR, then std, then 1."""
    mad = 1.4826 * (df - df.median()).abs().median()
    iqr = (df.quantile(0.75) - df.quantile(0.25)) / 1.349
    std = df.std().fillna(0.0)
    return mad.where(mad > 0, iqr.where(iqr > 0, std.where(std > 0, 1.0)))


def _one_hot_columns(table: pd.DataFrame, profile: DatasetProfile, params: dict[str, Any]) -> list[str]:
    spec = params.get("one_hot", False)
    if spec is True:
        max_categories = int(params.get("max_categories", 10))
        return [
            c.name for c in profile.columns
            if c.role is ColumnRole.CATEGORICAL and c.name in table.columns and c.n_unique <= max_categories
        ]
    if not spec:
        return []
    missing = [c for c in spec if c not in table.columns]
    if missing:
        raise ValueError(f"one_hot columns not in table: {missing}")
    return list(spec)


def _dummies(table: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(table[cols].astype(object).fillna("<missing>").astype(str), dtype=np.float64)


def _top_fraction(score: pd.Series, fraction: float) -> pd.Series:
    k = min(len(score), math.ceil(fraction * len(score)))
    return score.rank(method="first", ascending=False) <= k


def _reason(values: pd.Series, z: pd.Series) -> str:
    order = z.abs().sort_values(ascending=False, kind="stable")
    names = [n for i, n in enumerate(order.index[:2]) if i == 0 or order[n] >= 2.0]
    return ", ".join(f"{n} {_fmt(values[n])} ({_level(z[n])})" for n in names)


def _level(z: float) -> str:
    direction = "high" if z >= 0 else "low"
    return f"very {direction}" if abs(z) >= 5 else direction


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "missing"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
