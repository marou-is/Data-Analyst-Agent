import numpy as np
import pandas as pd
import pytest

from analyst.contracts import GROUP, AggregationSpec, DatasetSpec, MetricSpec, assert_clean_frame
from analyst.data.aggregate import aggregate, default_aggregation, snake_case
from analyst.data.clean import clean_table
from analyst.data.ingest import load_dataset
from analyst.data.profile import profile_dataset
from analyst.datasets import SUPERSTORE


@pytest.fixture
def table():
    return pd.DataFrame({
        "Row ID": range(1, 8),
        "Order ID": ["A", "A", "B", "C", "D", "D", "E"],
        "Order Date": pd.to_datetime([
            "2024-01-01 09:00", "2024-01-01 17:00", "2024-01-02", "2024-01-05",
            "2024-01-05", "2024-01-05", "2024-01-15",
        ], format="mixed"),
        "Postal Code": [2108, 2108, 90036, 1040, 10001, 10001, 98103],
        "Category": ["x", "x", "y", "x", "y", None, "y"],
        "Sales": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "Discount": [0.0, 0.2, 0.1, 0.0, 0.5, 0.3, 0.0],
    })


def metrics(spec):
    return {m.name: (m.agg, m.column) for m in spec.metrics}


def test_default_aggregation(table):
    spec = default_aggregation(profile_dataset(table), freq="W-MON")
    assert (spec.timestamp_col, spec.freq, spec.group_by) == ("Order Date", "W-MON", None)
    assert metrics(spec) == {
        "sales": ("sum", "Sales"),
        "discount": ("mean", "Discount"),
        "n_rows": ("count", None),
        "n_order_id": ("nunique", "Order ID"),
    }


def test_default_aggregation_needs_datetime():
    with pytest.raises(ValueError, match="datetime"):
        default_aggregation(profile_dataset(pd.DataFrame({"Sales": [1.0, 2.0, 3.0]})))


def test_snake_case():
    assert [snake_case(n) for n in ["Sales", "Sub-Category", "Unit Price ($)", "unitPrice", "Order ID"]] == [
        "sales", "sub_category", "unit_price", "unit_price", "order_id",
    ]


def test_daily_totals_and_gap_filling(table):
    spec = default_aggregation(profile_dataset(table))
    result = aggregate(table, spec)
    df = result.data
    assert_clean_frame(df, DatasetSpec(target_cols=result.metric_names))
    assert df.index.equals(pd.date_range("2024-01-01", "2024-01-15", freq="D", name="timestamp"))
    assert df["sales"].sum() == table["Sales"].sum()
    assert df["n_rows"].sum() == len(table)
    assert df.loc["2024-01-01", ["sales", "n_rows", "n_order_id"]].tolist() == [3.0, 2.0, 1.0]
    assert df.loc["2024-01-01", "discount"] == pytest.approx(0.1)
    assert df.loc["2024-01-03", ["sales", "n_rows", "n_order_id"]].tolist() == [0.0, 0.0, 0.0]
    assert np.isnan(df.loc["2024-01-03", "discount"])


@pytest.mark.parametrize(("freq", "labels"), [
    ("W-MON", ["2024-01-01", "2024-01-08", "2024-01-15"]),
    ("W", ["2023-12-31", "2024-01-07", "2024-01-14"]),
    ("MS", ["2024-01-01"]),
])
def test_period_start_labels(table, freq, labels):
    spec = AggregationSpec(timestamp_col="Order Date", freq=freq, metrics=[MetricSpec(name="sales", agg="sum", column="Sales")])
    df = aggregate(table, spec).data
    assert df.index.strftime("%Y-%m-%d").tolist() == labels
    assert df["sales"].sum() == table["Sales"].sum()


def test_group_by_long_format(table):
    spec = AggregationSpec(
        timestamp_col="Order Date", freq="W-MON", group_by="Category",
        metrics=[MetricSpec(name="sales", agg="sum", column="Sales"), MetricSpec(name="n_rows", agg="count")],
    )
    result = aggregate(table, spec)
    assert result.groups == ["(missing)", "x", "y"]
    assert len(result.data) == 3 * 3
    assert result.data.groupby(GROUP)["sales"].sum().to_dict() == {"(missing)": 6.0, "x": 7.0, "y": 15.0}
    y = result.frame("y")
    assert_clean_frame(y, DatasetSpec(target_cols=["sales", "n_rows"]))
    assert y["sales"].tolist() == [8.0, 0.0, 7.0]


def test_tolerates_string_timestamps_and_does_not_mutate(table):
    raw = table.assign(**{"Order Date": table["Order Date"].dt.strftime("%m/%d/%Y")})
    before = raw.copy()
    spec = default_aggregation(profile_dataset(table))
    result = aggregate(raw, spec)
    pd.testing.assert_frame_equal(raw, before)
    assert result.data["sales"].sum() == table["Sales"].sum()
    assert len(result.data) == 15


def test_unparseable_timestamps_are_ignored():
    raw = pd.DataFrame({"when": ["2024-01-01", "garbage", "2024-01-03"], "v": [1.0, 2.0, 3.0]})
    spec = AggregationSpec(timestamp_col="when", metrics=[MetricSpec(name="v", agg="sum", column="v")])
    assert aggregate(raw, spec).data["v"].tolist() == [1.0, 0.0, 3.0]


def test_missing_column_raises(table):
    spec = AggregationSpec(timestamp_col="Order Date", metrics=[MetricSpec(name="x", agg="sum", column="nope")])
    with pytest.raises(ValueError, match="nope"):
        aggregate(table, spec)


@pytest.fixture(scope="module")
def superstore():
    if not SUPERSTORE.path.exists():
        pytest.skip("Superstore data not downloaded")
    raw = load_dataset(SUPERSTORE.path)
    profile = profile_dataset(raw, name="superstore")
    return clean_table(raw, profile).data, default_aggregation(profile)


def test_superstore_default_spec(superstore):
    _, spec = superstore
    assert spec.timestamp_col == "Order Date"
    assert metrics(spec) == {
        "sales": ("sum", "Sales"), "quantity": ("sum", "Quantity"), "discount": ("mean", "Discount"),
        "profit": ("sum", "Profit"), "n_rows": ("count", None), "n_order_id": ("nunique", "Order ID"),
        "n_customer_id": ("nunique", "Customer ID"), "n_product_id": ("nunique", "Product ID"),
    }


def test_superstore_daily(superstore):
    table, spec = superstore
    result = aggregate(table, spec)
    df = result.data
    assert_clean_frame(df, DatasetSpec(target_cols=result.metric_names))
    assert len(df) == 1458
    assert (df.index.min(), df.index.max()) == (pd.Timestamp("2014-01-03"), pd.Timestamp("2017-12-30"))
    assert df["sales"].sum() == pytest.approx(table["Sales"].sum())
    assert df["profit"].sum() == pytest.approx(table["Profit"].sum())
    assert df["n_rows"].sum() == len(table) == 9993
    empty = df["n_rows"] == 0
    assert empty.any() and (df.loc[empty, "sales"] == 0).all() and df.loc[empty, "discount"].isna().all()


@pytest.mark.parametrize(("freq", "n_periods"), [("W-MON", 209), ("MS", 48)])
def test_superstore_weekly_monthly(superstore, freq, n_periods):
    table, spec = superstore
    df = aggregate(table, spec.model_copy(update={"freq": freq})).data
    assert len(df) == n_periods
    assert df["sales"].sum() == pytest.approx(table["Sales"].sum())
    assert df["n_rows"].sum() == len(table)


def test_superstore_by_category(superstore):
    table, spec = superstore
    result = aggregate(table, spec.model_copy(update={"group_by": "Category", "freq": "MS"}))
    assert result.groups == ["Furniture", "Office Supplies", "Technology"]
    assert len(result.data) == 48 * 3
    totals = result.data.groupby(GROUP)["sales"].sum()
    assert totals.sum() == pytest.approx(table["Sales"].sum())
    assert totals.to_dict() == pytest.approx(table.groupby("Category")["Sales"].sum().to_dict())
