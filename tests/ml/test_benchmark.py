import pandas as pd

from analyst.ml.benchmark import INJECTED_KINDS, inject_row_anomalies, precision_at_k, to_markdown


def _table(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "Sales": [float(i + 1) for i in range(n)],
        "Quantity": [1 + i % 5 for i in range(n)],
        "Discount": [0.1] * n,
        "Profit": [0.2 * (i + 1) for i in range(n)],
    })


def test_injection_is_deterministic_and_labelled():
    table = _table()
    a, kind_a = inject_row_anomalies(table, n=8, seed=3)
    b, kind_b = inject_row_anomalies(table, n=8, seed=3)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(kind_a, kind_b)
    assert len(a) == len(table) + 8
    assert a.index.is_unique
    pd.testing.assert_frame_equal(a.iloc[: len(table)], table)
    assert set(kind_a.iloc[len(table):]) == set(INJECTED_KINDS)
    assert (kind_a.iloc[: len(table)] == "").all()
    assert a["Quantity"].dtype == table["Quantity"].dtype
    injected = a[kind_a == "discount_profit"]
    assert (injected["Discount"] == 0.95).all() and (injected["Profit"] >= 1_500).all()


def test_precision_at_k():
    truth = pd.Series([True, False, True, False])
    score = pd.Series([0.9, 0.8, 0.7, 0.1])
    assert precision_at_k(truth, score, 2) == 0.5
    assert precision_at_k(truth, score, 0) == 0.0


def test_to_markdown():
    md = to_markdown(pd.DataFrame({"a": ["x"], "b": [0.12345]}))
    assert md.splitlines() == ["| a | b |", "|---|---|", "| x | 0.123 |"]
