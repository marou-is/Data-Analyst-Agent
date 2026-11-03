"""Shared contracts between the data, ml, agent and dashboard packages.

Owned by the coordinator. Workstreams import from this module but must not edit
it; request a change instead so every consumer stays in sync.

Frame conventions
-----------------
* A *raw frame* is whatever ``analyst.data.ingest.load_dataset`` returns: the
  file's columns as loaded, timestamp still in a regular column.
* A *clean frame* has a sorted, unique, tz-naive ``DatetimeIndex`` named
  ``timestamp`` and only the target columns, all float64. Everything downstream
  of ``analyst.data.clean.clean`` receives clean frames
  (see ``assert_clean_frame``).

Stage signatures (implemented by the workstreams)
-------------------------------------------------
* ``data.ingest.load_dataset(source, filename=None, sheet_name=0) -> pd.DataFrame``  raw frame (CSV/Excel/Parquet)
* ``data.ingest.infer_spec(raw, name) -> DatasetSpec``
* ``data.profile.profile_dataset(raw, name) -> DatasetProfile``   any table, includes quality
* ``data.quality.check_table_quality(raw, columns, n_duplicate_rows) -> QualityReport``
* ``data.quality.check_quality(raw, spec) -> QualityReport``     time-series checks
* ``data.clean.clean(raw, spec) -> CleanResult``
* ``data.profile.profile(clean_df, spec) -> DatasetProfile``     time-series profile
* ``ml.anomaly.detect_anomalies(clean_df, spec, target_col, config, labels=None) -> AnomalyResult``
* ``ml.forecast.forecast(clean_df, spec, target_col, config) -> ForecastResult``
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TIMESTAMP = "timestamp"

ANOMALY_COLUMNS = ("value", "score", "is_anomaly")
FORECAST_COLUMNS = ("yhat", "yhat_lower", "yhat_upper")
BACKTEST_COLUMNS = ("y", "yhat")


class _Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


def _check_frame(df: Any, columns: tuple[str, ...], what: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{what} must be a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{what} must have a DatetimeIndex")
    return df


def assert_clean_frame(df: pd.DataFrame, spec: DatasetSpec) -> None:
    """Raise ``ValueError`` if ``df`` violates the clean-frame convention."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("clean frame must have a DatetimeIndex")
    if df.index.name != TIMESTAMP:
        raise ValueError(f"clean frame index must be named {TIMESTAMP!r}, got {df.index.name!r}")
    if df.index.tz is not None:
        raise ValueError("clean frame index must be tz-naive")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("clean frame index must be sorted and unique")
    missing = [c for c in spec.target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"clean frame is missing target columns {missing}")
    non_float = [c for c in spec.target_cols if df[c].dtype != "float64"]
    if non_float:
        raise ValueError(f"target columns must be float64: {non_float}")


class DatasetSpec(_Model):
    name: str = "dataset"
    timestamp_col: str = TIMESTAMP
    """Name of the timestamp column in the *raw* frame."""
    target_cols: list[str] = Field(min_length=1)
    freq: str | None = None
    """Pandas offset alias such as ``"30min"`` or ``"D"``; inferred when ``None``."""
    season_length: int | None = Field(default=None, ge=2)
    """Observations per seasonal cycle (48 for daily seasonality at 30min)."""


class AnalysisConfig(_Model):
    anomaly_method: str = "seasonal_robust_zscore"
    anomaly_threshold: float = Field(default=3.5, gt=0)
    anomaly_params: dict[str, Any] = Field(default_factory=dict)
    forecast_model: str = "seasonal_naive"
    forecast_horizon: int = Field(default=48, gt=0)
    forecast_params: dict[str, Any] = Field(default_factory=dict)


class CleaningStep(_Model):
    name: str
    description: str
    rows_affected: int = Field(default=0, ge=0)


class CleanResult(_Model):
    data: pd.DataFrame
    steps: list[CleaningStep] = Field(default_factory=list)
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QualityIssue(_Model):
    check: str
    severity: Severity
    message: str
    column: str | None = None
    n_affected: int = Field(default=0, ge=0)


class QualityReport(_Model):
    issues: list[QualityIssue] = Field(default_factory=list)
    score: float = Field(ge=0, le=100)
    """100 means no issues found."""

    @property
    def passed(self) -> bool:
        return not any(i.severity is Severity.ERROR for i in self.issues)


class ColumnRole(str, Enum):
    """Semantic role of a column, independent of its storage dtype."""

    IDENTIFIER = "identifier"
    DATETIME = "datetime"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    CONSTANT = "constant"
    EMPTY = "empty"


class ValueCount(_Model):
    value: str
    count: int = Field(ge=0)


class ColumnProfile(_Model):
    name: str
    dtype: str
    n_missing: int = Field(ge=0)
    pct_missing: float = Field(ge=0, le=1)
    n_unique: int = Field(ge=0)
    role: ColumnRole | None = None
    pct_unique: float | None = Field(default=None, ge=0, le=1)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    max: float | None = None
    n_zeros: int | None = None
    n_negative: int | None = None
    n_outliers: int | None = None
    """Values outside 1.5 * IQR of the quartiles (numeric columns only)."""
    min_date: datetime | None = None
    max_date: datetime | None = None
    top_values: list[ValueCount] = Field(default_factory=list)


class DatasetProfile(_Model):
    """Structured, JSON-serialisable description of a dataset.

    Small enough to be passed to an LLM as context and stored in analysis history.
    """

    name: str = "dataset"
    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    n_duplicate_rows: int = Field(default=0, ge=0)
    memory_mb: float | None = None
    start: datetime | None = None
    end: datetime | None = None
    inferred_freq: str | None = None
    columns: list[ColumnProfile] = Field(default_factory=list)
    quality: QualityReport | None = None

    def column(self, name: str) -> ColumnProfile:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def columns_with_role(self, role: ColumnRole) -> list[str]:
        return [c.name for c in self.columns if c.role is role]


class AnomalyResult(_Model):
    method: str
    target_col: str
    frame: pd.DataFrame
    """Indexed like the clean frame; columns ``value``, ``score`` (higher is more
    anomalous), ``is_anomaly`` (bool)."""
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    """Evaluation against labels when available (precision, recall, f1, ...)."""

    @field_validator("frame")
    @classmethod
    def _frame_shape(cls, df: Any) -> pd.DataFrame:
        df = _check_frame(df, ANOMALY_COLUMNS, "AnomalyResult.frame")
        if df["is_anomaly"].dtype != bool:
            raise ValueError("AnomalyResult.frame['is_anomaly'] must be bool")
        return df

    @property
    def n_anomalies(self) -> int:
        return int(self.frame["is_anomaly"].sum())


class ForecastResult(_Model):
    model: str
    target_col: str
    horizon: int = Field(gt=0)
    forecast: pd.DataFrame
    """Future timestamps; columns ``yhat``, ``yhat_lower``, ``yhat_upper``."""
    backtest: pd.DataFrame
    """Held-out history; columns ``y`` (actual) and ``yhat`` (predicted)."""
    metrics: dict[str, float] = Field(default_factory=dict)
    """Backtest metrics of ``model`` (mae, rmse, mape, ...)."""
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    """Same metrics for the naive baseline on the same backtest window."""
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("forecast")
    @classmethod
    def _forecast_shape(cls, df: Any) -> pd.DataFrame:
        return _check_frame(df, FORECAST_COLUMNS, "ForecastResult.forecast")

    @field_validator("backtest")
    @classmethod
    def _backtest_shape(cls, df: Any) -> pd.DataFrame:
        return _check_frame(df, BACKTEST_COLUMNS, "ForecastResult.backtest")

    @model_validator(mode="after")
    def _horizon_matches(self) -> ForecastResult:
        if len(self.forecast) != self.horizon:
            raise ValueError(
                f"forecast has {len(self.forecast)} rows but horizon is {self.horizon}"
            )
        return self


class Insight(_Model):
    title: str
    detail: str
    severity: Severity = Severity.INFO
    source: Literal["profile", "quality", "anomaly", "forecast"]


class AnalysisReport(_Model):
    spec: DatasetSpec
    config: AnalysisConfig
    quality: QualityReport
    cleaning: CleanResult
    profile: DatasetProfile
    anomalies: list[AnomalyResult] = Field(default_factory=list)
    forecasts: list[ForecastResult] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    """Non-fatal stage failures, e.g. a forecast model that raised."""
    created_at: datetime = Field(default_factory=datetime.now)


def validate_report(report: AnalysisReport, allow_warnings: bool = False) -> None:
    """Cross-object consistency checks the dashboard relies on; raises ``ValueError``."""
    assert_clean_frame(report.cleaning.data, report.spec)
    index = report.cleaning.data.index
    for result in report.anomalies:
        if not result.frame.index.equals(index):
            raise ValueError(f"anomaly frame for {result.target_col!r} is not aligned with the clean frame")
    for result in report.forecasts:
        if not result.forecast.index.min() > index.max():
            raise ValueError(f"forecast for {result.target_col!r} does not start after the history")
    expected = set(report.spec.target_cols)
    if {r.target_col for r in report.anomalies} != expected and not allow_warnings:
        raise ValueError("missing anomaly results for some targets")
    if {r.target_col for r in report.forecasts} != expected and not allow_warnings:
        raise ValueError("missing forecast results for some targets")
    if report.warnings and not allow_warnings:
        raise ValueError(f"report has stage warnings: {report.warnings}")
    if not report.insights:
        raise ValueError("report has no insights")
