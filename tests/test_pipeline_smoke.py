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


@pytest.mark.skipif(not NYC_TAXI.path.exists(), reason="NYC taxi data not downloaded")
def test_nyc_taxi_end_to_end():
    report = run_public_dataset("nyc_taxi")
    check_report(report)
    assert report.profile.n_rows == 10320
    assert set(report.anomalies[0].metrics) >= {"precision", "recall", "f1", "event_recall"}
