import numpy as np
import pandas as pd
import pytest

from analyst.contracts import TIMESTAMP, AnalysisConfig, AnomalyResult, DatasetSpec
from analyst.data.clean import clean
from analyst.data.ingest import load_dataset
from analyst.datasets import NYC_TAXI, label_series
from analyst.ml.anomaly import METHODS, detect_anomalies, robust_zscore


def _run(df, spec, method, labels=None, **kwargs):
    return detect_anomalies(df, spec, "value", AnalysisConfig(anomaly_method=method, **kwargs), labels=labels)


def _daily_sales(n_days: int = 140, seed: int = 0) -> tuple[pd.DataFrame, DatasetSpec]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_days, freq="D", name=TIMESTAMP)
    weekly = np.where(index.dayofweek >= 5, 40.0, 0.0)
    value = 200.0 + 0.2 * np.arange(n_days) + weekly + rng.normal(0, 5, n_days)
    value[min(100, n_days - 1)] += 120.0
    spec = DatasetSpec(name="daily", target_cols=["value"], freq="D", season_length=7)
    return pd.DataFrame({"value": value}, index=index), spec


@pytest.mark.parametrize("method", METHODS)
def test_methods_return_valid_aligned_result(clean_frame, synthetic, method):
    _, spec, windows = synthetic
    labels = label_series(clean_frame.index, windows)
    result = _run(clean_frame, spec, method, labels=labels)
    assert isinstance(result, AnomalyResult)
    assert result.method == method
    assert result.frame.index.equals(clean_frame.index)
    assert result.frame["score"].dtype == np.float64
    assert np.isfinite(result.frame["score"]).all()
    assert result.metrics["event_recall"] == 1.0
    assert 0 < result.n_anomalies < 0.1 * len(clean_frame)


@pytest.mark.parametrize("method", METHODS)
def test_methods_are_deterministic(clean_frame, synthetic, method):
    spec = synthetic[1]
    a = _run(clean_frame, spec, method).frame
    b = _run(clean_frame, spec, method).frame
    pd.testing.assert_frame_equal(a, b)


def test_default_method_unchanged(clean_frame, synthetic):
    spec = synthetic[1]
    result = detect_anomalies(clean_frame, spec, "value", AnalysisConfig())
    assert result.method == "seasonal_robust_zscore"
    y = clean_frame["value"]
    expected = robust_zscore(y - y.shift(spec.season_length))
    np.testing.assert_allclose(result.frame["score"], expected)
    assert result.params == {"season_length": 48, "threshold": 3.5}


@pytest.mark.parametrize("method", ["stl_robust_zscore", "mstl_robust_zscore"])
def test_threshold_is_respected(clean_frame, synthetic, method):
    spec = synthetic[1]
    low = _run(clean_frame, spec, method, anomaly_threshold=3.0).frame["is_anomaly"]
    high = _run(clean_frame, spec, method, anomaly_threshold=6.0).frame["is_anomaly"]
    assert high.sum() < low.sum()
    assert not (high & ~low).any()


@pytest.mark.parametrize("method", ["stl_robust_zscore", "mstl_robust_zscore"])
def test_daily_series_with_weekly_season(method):
    df, spec = _daily_sales()
    result = _run(df, spec, method)
    assert result.frame["score"].idxmax() == df.index[100]
    assert result.frame["is_anomaly"].iloc[100]
    if method == "mstl_robust_zscore":
        assert result.params["periods"] == [7]


def test_mstl_adds_weekly_period_only_for_long_subdaily_series(clean_frame, synthetic):
    spec = synthetic[1]
    assert _run(clean_frame, spec, "mstl_robust_zscore").params["periods"] == [48]
    explicit = _run(clean_frame, spec, "mstl_robust_zscore", anomaly_params={"periods": [48, 336]})
    assert explicit.params["periods"] == [48, 336]


@pytest.mark.parametrize("method", ["stl_robust_zscore", "mstl_robust_zscore"])
def test_short_series_raises_clear_error(method):
    df, spec = _daily_sales(n_days=10)
    with pytest.raises(ValueError, match="at least two full seasons"):
        _run(df, spec, method)


@pytest.mark.parametrize("method", ["stl_robust_zscore", "mstl_robust_zscore"])
def test_missing_season_length_raises(method):
    df, spec = _daily_sales()
    with pytest.raises(ValueError, match="season_length"):
        _run(df, spec.model_copy(update={"season_length": None}), method)


def test_stl_tolerates_missing_values():
    df, spec = _daily_sales()
    df.iloc[[5, 6, 50]] = np.nan
    result = _run(df, spec, "stl_robust_zscore")
    assert np.isfinite(result.frame["score"]).all()
    assert result.frame["is_anomaly"].iloc[100]


def test_score_smoothing_window(clean_frame, synthetic):
    spec = synthetic[1]
    result = _run(clean_frame, spec, "seasonal_robust_zscore", anomaly_params={"window": 5})
    assert result.params["window"] == 5
    with pytest.raises(ValueError, match="window"):
        _run(clean_frame, spec, "seasonal_robust_zscore", anomaly_params={"window": 0})


def test_unknown_method_lists_available(clean_frame, synthetic):
    with pytest.raises(ValueError, match="mstl_robust_zscore"):
        _run(clean_frame, synthetic[1], "prophet")


@pytest.fixture(scope="module")
def nyc():
    df = clean(load_dataset(NYC_TAXI.path), NYC_TAXI.spec).data
    return df, label_series(df.index, NYC_TAXI.anomaly_windows)


def test_weekly_aware_decompositions_beat_baseline_on_nyc_taxi(nyc):
    """At the default threshold 3.5 the baseline F1 is ~0.20. STL with the daily period
    alone only ties it (weekly effects stay in the residual); adding the weekly cycle
    (MSTL, or STL with a weekly period) roughly doubles F1."""
    df, labels = nyc
    spec = NYC_TAXI.spec
    f1 = {
        name: _run(df, spec, method, labels=labels, anomaly_params=params).metrics["f1"]
        for name, method, params in [
            ("baseline", "seasonal_robust_zscore", {}),
            ("stl_daily", "stl_robust_zscore", {}),
            ("stl_weekly", "stl_robust_zscore", {"season_length": 336}),
            ("mstl", "mstl_robust_zscore", {}),
        ]
    }
    assert f1["baseline"] == pytest.approx(0.201, abs=0.005)
    assert abs(f1["stl_daily"] - f1["baseline"]) < 0.03
    assert f1["stl_weekly"] > f1["baseline"] + 0.1
    assert f1["mstl"] > f1["baseline"] + 0.15
