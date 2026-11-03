"""End-to-end gate: must pass before and after every workstream integration."""

import pytest

from analyst.agent import run_analysis, run_public_dataset
from analyst.contracts import validate_report as check_report
from analyst.datasets import NYC_TAXI


def test_synthetic_end_to_end(synthetic):
    raw, spec, windows = synthetic
    report = run_analysis(raw, spec=spec, anomaly_windows=windows)
    check_report(report)
    assert report.quality.issues, "synthetic data has injected missing values and duplicates"
    assert report.cleaning.rows_after < report.cleaning.rows_before
    assert report.anomalies[0].metrics["event_recall"] > 0


def test_infers_spec_without_hints(synthetic):
    raw, _, _ = synthetic
    report = run_analysis(raw)
    check_report(report)
    assert report.spec.timestamp_col == "timestamp"
    assert report.spec.freq == "30min"
    assert report.spec.season_length == 48


def test_superstore_is_aggregated_not_deduplicated():
    from analyst.contracts import AnalysisConfig
    from analyst.datasets import SUPERSTORE

    report = run_analysis(SUPERSTORE.path)
    check_report(report)
    assert report.aggregation is not None
    assert report.spec.target_cols == ["sales", "quantity", "discount", "profit"]
    assert report.profile.n_rows == 1458
    assert report.cleaning.rows_before == 9994
    assert report.cleaning.data["sales"].sum() == pytest.approx(2296919.4883)
    assert report.table_profile.n_rows == 9994

    rows = report.row_anomalies
    assert rows is not None and rows.method == "isolation_forest"
    assert len(rows.frame) == 9993 and rows.n_anomalies == 100
    assert "Profit/Sales" in rows.features
    assert (rows.frame["reason"] != "").sum() == rows.n_anomalies
    assert any("unusual rows" in i.title for i in report.insights)

    monthly = run_analysis(SUPERSTORE.path, config=AnalysisConfig(aggregation_freq="MS", forecast_horizon=6))
    check_report(monthly)
    assert monthly.profile.n_rows == 48


@pytest.mark.parametrize(
    ("freq", "season"),
    [("30min", 48), ("h", 24), ("D", 7), ("W", 52), ("W-MON", 52), ("MS", 12), ("ME", 12), ("QS", 4), ("7min", None), ("YS", None)],
)
def test_infer_season_length(freq, season):
    from analyst.agent.pipeline import infer_season_length

    assert infer_season_length(freq) == season


def test_synthetic_with_few_duplicates_is_not_aggregated(synthetic):
    raw, spec, _ = synthetic
    report = run_analysis(raw, spec=spec)
    assert report.aggregation is None and report.row_anomalies is None


def test_anomaly_method_is_chosen_from_data(synthetic):
    from analyst.contracts import AnalysisConfig

    raw, spec, _ = synthetic
    assert run_analysis(raw, spec=spec).anomalies[0].method == "seasonal_robust_zscore"  # only 4 weeks
    explicit = run_analysis(raw, spec=spec, config=AnalysisConfig(anomaly_method="stl_robust_zscore"))
    assert explicit.anomalies[0].method == "stl_robust_zscore"


def test_margin_ratio_detection():
    import pandas as pd

    from analyst.agent.pipeline import margin_ratio
    from analyst.data.profile import profile_dataset

    both = profile_dataset(pd.DataFrame({"Revenue": [1.0, 5.0, 3.0], "Net Profit": [0.5, -1.0, 2.0]}))
    assert margin_ratio(both) == ["Net Profit", "Revenue"]
    assert margin_ratio(profile_dataset(pd.DataFrame({"Sales": [1.0, 5.0, 3.0]}))) is None


def test_profile_source_accepts_upload_bytes():
    import io

    from analyst.agent import profile_source
    from analyst.datasets import SUPERSTORE

    p = profile_source(io.BytesIO(SUPERSTORE.path.read_bytes()), filename="superstore.csv")
    assert (p.name, p.n_rows, p.n_cols) == ("superstore", 9994, 21)


@pytest.mark.skipif(not NYC_TAXI.path.exists(), reason="NYC taxi data not downloaded")
def test_nyc_taxi_end_to_end():
    report = run_public_dataset("nyc_taxi")
    check_report(report)
    assert report.profile.n_rows == 10320
    assert set(report.anomalies[0].metrics) >= {"precision", "recall", "f1", "event_recall"}
    assert report.anomalies[0].method == "mstl_robust_zscore"
    assert report.anomalies[0].metrics["f1"] > 0.4


def test_aggregated_business_data_keeps_baseline_detector():
    from analyst.datasets import SUPERSTORE

    assert run_analysis(SUPERSTORE.path).anomalies[0].method == "seasonal_robust_zscore"
