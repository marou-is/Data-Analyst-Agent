import math

import numpy as np
import pandas as pd
import pytest

from analyst.contracts import AnalysisConfig, ColumnProfile, ColumnRole, DatasetProfile, RowAnomalyResult
from analyst.data.ingest import load_dataset
from analyst.data.profile import profile_dataset
from analyst.datasets import SUPERSTORE
from analyst.ml.benchmark import inject_row_anomalies
from analyst.ml.row_anomaly import METHODS, detect_row_anomalies

MARGIN = [["Profit", "Sales"]]


def _config(method="isolation_forest", contamination=0.01, **params):
    return AnalysisConfig(row_anomaly_method=method, row_anomaly_contamination=contamination, row_anomaly_params=params)


def _profile(table: pd.DataFrame, roles: dict[str, ColumnRole]) -> DatasetProfile:
    columns = [
        ColumnProfile(name=c, dtype=str(table[c].dtype), n_missing=int(table[c].isna().sum()),
                      pct_missing=float(table[c].isna().mean()), n_unique=int(table[c].nunique()), role=role)
        for c, role in roles.items()
    ]
    return DatasetProfile(name="toy", n_rows=len(table), n_cols=table.shape[1], columns=columns)


def _toy(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, DatasetProfile]:
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({
        "id": np.arange(n),
        "price": rng.normal(100, 10, n),
        "qty": rng.integers(1, 10, n).astype(float),
        "shop": rng.choice(["a", "b", "c"], n),
        "note": [f"row {i}" for i in range(n)],
    }, index=pd.RangeIndex(1000, 1000 + n))
    table.loc[1005, "price"] = 1_000.0
    roles = {"id": ColumnRole.IDENTIFIER, "price": ColumnRole.NUMERIC, "qty": ColumnRole.NUMERIC,
             "shop": ColumnRole.CATEGORICAL, "note": ColumnRole.TEXT}
    return table, _profile(table, roles)


@pytest.fixture(scope="module")
def superstore():
    raw = load_dataset(SUPERSTORE.path)
    profile = profile_dataset(raw, name=SUPERSTORE.name)
    table, kind = inject_row_anomalies(raw, n=50, seed=0)
    return table, profile, kind


@pytest.fixture(scope="module")
def superstore_result(superstore):
    table, profile, _ = superstore
    return detect_row_anomalies(table, profile, _config())


def _check_contract(result: RowAnomalyResult, table: pd.DataFrame) -> None:
    assert isinstance(result, RowAnomalyResult)
    frame = result.frame
    assert frame.index.equals(table.index)
    assert frame["is_anomaly"].dtype == bool
    assert frame["score"].dtype == np.float64
    assert np.isfinite(frame["score"]).all()
    has_reason = frame["reason"].str.len() > 0
    assert (has_reason == frame["is_anomaly"]).all()


def test_superstore_contract_and_features(superstore, superstore_result):
    table, _, _ = superstore
    _check_contract(superstore_result, table)
    assert superstore_result.method == "isolation_forest"
    assert superstore_result.features == ["Sales", "Quantity", "Discount", "Profit"]
    assert superstore_result.n_anomalies == math.ceil(0.01 * len(table))
    flagged = superstore_result.frame["score"][superstore_result.frame["is_anomaly"]]
    assert flagged.min() >= superstore_result.frame["score"][~superstore_result.frame["is_anomaly"]].max()


def test_isolation_forest_recovers_injected_rows(superstore, superstore_result):
    """Benchmark (``python -m analyst.ml.benchmark``): recall 0.94 at contamination 0.01
    with default params and 1.00 with the Profit/Sales margin ratio; thresholds below
    leave headroom. The injection is synthetic, so this checks plausibility, not real-world recall."""
    table, profile, kind = superstore
    truth = kind != ""
    flags = superstore_result.frame["is_anomaly"]
    assert flags[truth].mean() >= 0.85
    with_margin = detect_row_anomalies(table, profile, _config(ratios=MARGIN)).frame["is_anomaly"]
    assert with_margin[truth].mean() >= 0.95
    baseline = detect_row_anomalies(table, profile, _config("robust_zscore")).frame["is_anomaly"]
    assert flags[truth].mean() > baseline[truth].mean()


def test_reason_names_most_deviating_feature(superstore, superstore_result):
    _, _, kind = superstore
    frame = superstore_result.frame
    huge = frame.loc[(kind == "huge_sales") & frame["is_anomaly"], "reason"]
    assert len(huge) > 0
    assert huge.str.match(r"^Sales [\d,]+\.\d\d \(very high\)").all()
    discount = frame.loc[(kind == "discount_profit") & frame["is_anomaly"], "reason"]
    assert discount.str.contains("Profit").all()


def test_is_deterministic(superstore, superstore_result):
    table, profile, _ = superstore
    again = detect_row_anomalies(table, profile, _config())
    pd.testing.assert_frame_equal(again.frame, superstore_result.frame)


def test_one_hot_uses_low_cardinality_categoricals(superstore):
    table, profile, _ = superstore
    sample = table.iloc[:2000]
    result = detect_row_anomalies(sample, profile, _config(one_hot=True))
    assert set(result.params["one_hot"]) == {"Ship Mode", "Segment", "Region", "Category"}
    assert result.features == ["Sales", "Quantity", "Discount", "Profit"]
    _check_contract(result, sample)


@pytest.mark.parametrize("method", METHODS)
def test_toy_table_contract_and_reason(method):
    table, profile = _toy()
    result = detect_row_anomalies(table, profile, _config(method, contamination=0.01))
    _check_contract(result, table)
    assert result.features == ["price", "qty"]
    assert result.n_anomalies == 2
    assert result.frame["score"].idxmax() == 1005
    assert result.frame.loc[1005, "reason"].startswith("price 1,000 (very high)")


def test_ratio_feature_and_reason():
    table, profile = _toy()
    table.loc[1010, ["price", "qty"]] = [100.0, 0.0]
    result = detect_row_anomalies(table, profile, _config(ratios=[["price", "qty"]]))
    assert result.features == ["price", "qty", "price/qty"]
    assert result.params["ratios"] == [["price", "qty"]]
    assert np.isfinite(result.frame["score"]).all()


def test_missing_values_are_imputed():
    table, profile = _toy()
    table.loc[table.index[:20], "qty"] = np.nan
    table.loc[table.index[30:35], "price"] = np.nan
    for method in METHODS:
        result = detect_row_anomalies(table, profile, _config(method, contamination=0.05))
        _check_contract(result, table)
        assert result.frame.loc[1005, "is_anomaly"]


def test_single_numeric_feature():
    table, profile = _toy()
    profile = _profile(table, {"price": ColumnRole.NUMERIC, "qty": ColumnRole.CATEGORICAL})
    result = detect_row_anomalies(table, profile, _config())
    assert result.features == ["price"]
    assert result.frame["is_anomaly"].loc[1005]
    _check_contract(result, table)


def test_no_numeric_features_returns_all_normal():
    table, _ = _toy()
    profile = _profile(table, {"id": ColumnRole.IDENTIFIER, "shop": ColumnRole.CATEGORICAL})
    result = detect_row_anomalies(table, profile, _config())
    assert result.n_anomalies == 0
    assert result.features == []
    assert result.params["skipped"] == "no numeric features"
    _check_contract(result, table)


def test_robust_zscore_explicit_threshold():
    table, profile = _toy()
    result = detect_row_anomalies(table, profile, _config("robust_zscore", threshold=10.0))
    assert result.frame["is_anomaly"].sum() == 1
    assert result.frame.loc[1005, "is_anomaly"]


def test_invalid_options_raise():
    table, profile = _toy()
    with pytest.raises(ValueError, match="isolation_forest"):
        detect_row_anomalies(table, profile, _config("lof"))
    with pytest.raises(ValueError, match="not in table"):
        detect_row_anomalies(table, profile, _config(ratios=[["price", "missing"]]))
    with pytest.raises(ValueError, match="transform"):
        detect_row_anomalies(table, profile, _config(transform="log"))
