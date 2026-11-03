"""Benchmark of the anomaly detectors; run ``python -m analyst.ml.benchmark``.

* Time series: every method plus a threshold sweep on the labelled NYC taxi windows
  and on the synthetic series with injected windows.
* Rows: Superstore with ~50 injected implausible order lines (fixed seed).

Forecasting is out of scope for this milestone. The NYC labels are the only real
ground truth, so any threshold chosen from this table is tuned on the test labels.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from analyst.contracts import AnalysisConfig, DatasetSpec
from analyst.data.clean import clean
from analyst.data.ingest import load_dataset
from analyst.data.profile import profile_dataset
from analyst.datasets import NYC_TAXI, SUPERSTORE, label_series
from analyst.ml.anomaly import detect_anomalies
from analyst.ml.evaluation import classification_metrics
from analyst.ml.row_anomaly import detect_row_anomalies
from analyst.synthetic import make_synthetic_series

THRESHOLDS = (3.0, 3.5, 4.0, 5.0)
CONTAMINATIONS = (0.005, 0.01, 0.02)
INJECTED_KINDS = ("huge_sales", "discount_profit", "negative_margin", "bulk_quantity")


def ts_variants(season: int) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("seasonal_robust_zscore", {}),
        ("stl_robust_zscore", {}),
        ("stl_robust_zscore", {"season_length": 7 * season}),
        ("mstl_robust_zscore", {}),
        ("mstl_robust_zscore", {"window": season}),
    ]


MARGIN = [["Profit", "Sales"]]
ROW_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("robust_zscore", {}),
    ("robust_zscore", {"ratios": MARGIN}),
    ("isolation_forest", {"transform": "none", "max_samples": 256}),
    ("isolation_forest", {}),
    ("isolation_forest", {"ratios": MARGIN}),
    ("isolation_forest", {"ratios": MARGIN, "one_hot": True}),
]


def inject_row_anomalies(table: pd.DataFrame, n: int = 50, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Append ``n`` implausible copies of random Superstore rows.

    Returns ``(table, kind)`` where ``kind`` names the injected defect per row
    (one of ``INJECTED_KINDS``) and is ``""`` for original rows. Kinds:
    Sales far above the observed maximum, Discount 0.95 with a large positive Profit,
    no discount with a margin of -300..-600%, and bulk Quantity at an unchanged price.
    """
    rng = np.random.default_rng(seed)
    fake = table.iloc[rng.choice(len(table), size=n, replace=False)].copy()
    kinds = np.array([INJECTED_KINDS[i % len(INJECTED_KINDS)] for i in range(n)])
    sales = fake["Sales"].to_numpy(dtype=float)
    profit = fake["Profit"].to_numpy(dtype=float).copy()
    discount = fake["Discount"].to_numpy(dtype=float).copy()
    quantity = fake["Quantity"].to_numpy().copy()

    m = kinds == "huge_sales"
    sales = np.where(m, rng.uniform(40_000, 100_000, n), sales)
    profit = np.where(m, sales * rng.uniform(0.05, 0.2, n), profit)
    m = kinds == "discount_profit"
    discount = np.where(m, 0.95, discount)
    profit = np.where(m, rng.uniform(1_500, 5_000, n), profit)
    m = kinds == "negative_margin"
    discount = np.where(m, 0.0, discount)
    profit = np.where(m, -sales * rng.uniform(3, 6, n), profit)
    m = kinds == "bulk_quantity"
    quantity = np.where(m, rng.integers(40, 120, n), quantity)

    fake = fake.assign(Sales=sales, Profit=profit, Discount=discount, Quantity=quantity.astype(table["Quantity"].dtype))
    start = int(table.index.max()) + 1
    fake.index = pd.RangeIndex(start, start + n)
    out = pd.concat([table, fake])
    kind = pd.Series("", index=out.index, name="kind", dtype=object)
    kind.loc[fake.index] = kinds
    return out, kind.astype(str)


def run_timeseries_benchmark() -> pd.DataFrame:
    rows = []
    for name, df, spec, labels in _labelled_series():
        for method, params in ts_variants(spec.season_length or 1):
            t0 = time.perf_counter()
            result = detect_anomalies(df, spec, "value", AnalysisConfig(anomaly_method=method, anomaly_params=params))
            seconds = time.perf_counter() - t0
            for threshold in THRESHOLDS:
                flags = result.frame["score"] > threshold
                rows.append({
                    "dataset": name, "method": method, "params": _fmt_params(params), "threshold": threshold,
                    **classification_metrics(labels, flags), "pct_flagged": float(flags.mean()), "seconds": seconds,
                })
    return pd.DataFrame(rows)


def run_row_benchmark(n_injected: int = 50, seed: int = 0) -> pd.DataFrame:
    raw = load_dataset(SUPERSTORE.path)
    profile = profile_dataset(raw, name=SUPERSTORE.name)
    table, kind = inject_row_anomalies(raw, n=n_injected, seed=seed)
    truth = kind != ""
    rows = []
    for method, params in ROW_VARIANTS:
        scores = None
        for contamination in CONTAMINATIONS:
            config = AnalysisConfig(row_anomaly_method=method, row_anomaly_contamination=contamination,
                                    row_anomaly_params=params)
            t0 = time.perf_counter()
            result = detect_row_anomalies(table, profile, config)
            seconds = time.perf_counter() - t0
            scores = result.frame["score"]
            flags = result.frame["is_anomaly"]
            m = classification_metrics(truth, flags)
            per_kind = {f"recall_{k}": float(flags[kind == k].mean()) for k in INJECTED_KINDS}
            rows.append({
                "method": method, "params": _fmt_params(params), "contamination": contamination,
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "precision_at_k": precision_at_k(truth, scores, int(truth.sum())),
                **per_kind, "pct_flagged": float(flags.mean()), "seconds": seconds,
            })
    return pd.DataFrame(rows)


def run_benchmark() -> pd.DataFrame:
    """Both benchmarks in one long frame; ``section`` is ``"timeseries"`` or ``"rows"``."""
    return pd.concat(
        [run_timeseries_benchmark().assign(section="timeseries"), run_row_benchmark().assign(section="rows")],
        ignore_index=True,
    )


def precision_at_k(truth: pd.Series, score: pd.Series, k: int) -> float:
    """Share of true anomalies among the ``k`` highest scores."""
    if k <= 0:
        return 0.0
    top = score.rank(method="first", ascending=False) <= k
    return float(truth[top].mean())


def _labelled_series() -> list[tuple[str, pd.DataFrame, DatasetSpec, pd.Series]]:
    out = []
    df = clean(load_dataset(NYC_TAXI.path), NYC_TAXI.spec).data
    out.append(("nyc_taxi", df, NYC_TAXI.spec, label_series(df.index, NYC_TAXI.anomaly_windows)))
    raw, spec, windows = make_synthetic_series()
    df = clean(raw, spec).data
    out.append(("synthetic", df, spec, label_series(df.index, windows)))
    return out


def _fmt_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "-"


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def cell(v: Any) -> str:
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    t0 = time.perf_counter()
    ts = run_timeseries_benchmark()
    for name, part in ts.groupby("dataset", sort=False):
        print(f"\n### Time-series anomalies: {name}\n")
        print(to_markdown(part.drop(columns="dataset")))
    rows = run_row_benchmark()
    print("\n### Row anomalies: Superstore + 50 injected rows\n")
    print(to_markdown(rows))
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
