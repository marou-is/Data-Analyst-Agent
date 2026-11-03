import pandas as pd
import pytest
from pydantic import ValidationError

from analyst.contracts import (
    AggregateResult,
    AggregationSpec,
    AnomalyResult,
    MetricSpec,
    RowAnomalyResult,
    DatasetSpec,
    ForecastResult,
    QualityIssue,
    QualityReport,
    Severity,
    assert_clean_frame,
)

IDX = pd.date_range("2024-01-01", periods=4, freq="h", name="timestamp")


def test_spec_requires_targets():
    with pytest.raises(ValidationError):
        DatasetSpec(target_cols=[])


def test_anomaly_frame_must_have_columns_and_bool_flag():
    good = pd.DataFrame({"value": 1.0, "score": 0.0, "is_anomaly": False}, index=IDX)
    assert AnomalyResult(method="m", target_col="value", frame=good).n_anomalies == 0
    with pytest.raises(ValidationError):
        AnomalyResult(method="m", target_col="value", frame=good.drop(columns="score"))
    with pytest.raises(ValidationError):
        AnomalyResult(method="m", target_col="value", frame=good.assign(is_anomaly=0))


def test_forecast_length_must_match_horizon():
    fc = pd.DataFrame({"yhat": 1.0, "yhat_lower": 0.0, "yhat_upper": 2.0}, index=IDX)
    bt = pd.DataFrame({"y": 1.0, "yhat": 1.0}, index=IDX)
    ForecastResult(model="m", target_col="value", horizon=4, forecast=fc, backtest=bt)
    with pytest.raises(ValidationError):
        ForecastResult(model="m", target_col="value", horizon=3, forecast=fc, backtest=bt)


def test_quality_report_passed():
    warn = QualityIssue(check="c", severity=Severity.WARNING, message="m")
    err = QualityIssue(check="c", severity=Severity.ERROR, message="m")
    assert QualityReport(issues=[warn], score=90).passed
    assert not QualityReport(issues=[warn, err], score=60).passed


def test_metric_spec_requires_column_unless_count():
    MetricSpec(name="n_rows", agg="count")
    with pytest.raises(ValidationError):
        MetricSpec(name="sales", agg="sum")


def test_aggregate_result_shapes():
    spec = AggregationSpec(timestamp_col="Order Date", metrics=[MetricSpec(name="sales", agg="sum", column="Sales")])
    flat = pd.DataFrame({"sales": [1.0, 2.0, 3.0, 4.0]}, index=IDX)
    assert AggregateResult(spec=spec, data=flat).frame() is flat
    with pytest.raises(ValidationError):
        AggregateResult(spec=spec, data=flat.astype("int64"))

    grouped_spec = spec.model_copy(update={"group_by": "Region"})
    grouped = pd.DataFrame({"sales": [1.0, 2.0, 3.0, 4.0], "group": ["E", "W", "E", "W"]}, index=IDX)
    result = AggregateResult(spec=grouped_spec, data=grouped, groups=["E", "W"])
    assert result.frame("W")["sales"].tolist() == [2.0, 4.0]
    with pytest.raises(ValidationError):
        AggregateResult(spec=grouped_spec, data=flat)


def test_row_anomaly_frame():
    frame = pd.DataFrame({"score": [0.1, 0.9], "is_anomaly": [False, True], "reason": ["", "discount 80%"]})
    assert RowAnomalyResult(method="m", frame=frame).n_anomalies == 1
    with pytest.raises(ValidationError):
        RowAnomalyResult(method="m", frame=frame.drop(columns="reason"))


def test_assert_clean_frame():
    spec = DatasetSpec(target_cols=["value"])
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=IDX)
    assert_clean_frame(df, spec)
    with pytest.raises(ValueError):
        assert_clean_frame(df.iloc[::-1], spec)
    with pytest.raises(ValueError):
        assert_clean_frame(df.rename_axis("ts"), spec)
    with pytest.raises(ValueError):
        assert_clean_frame(df.astype("int64"), spec)
