import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from analyst.agent import load_source, profile_source
from analyst.contracts import ColumnRole
from analyst.dashboard.charts import (
    column_chart,
    missing_values_chart,
    numeric_distribution_chart,
    records_per_month_chart,
    role_breakdown_chart,
    to_datetime,
    top_values_chart,
)
from analyst.datasets import SUPERSTORE


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(0)
    n = 60
    raw = pd.DataFrame(
        {
            "order_id": [f"A{i:04d}" for i in range(n)],
            "date": pd.date_range("2024-01-01", periods=n, freq="5D").strftime("%Y-%m-%d"),
            "amount": np.append(rng.normal(100, 10, n - 1), 1000.0),
            "segment": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice(["yes", "no"], n),
            "country": "US",
            "note": [None if i % 4 == 0 else f"free text comment number {i} with extra words" for i in range(n)],
            "blank": [None] * n,
        }
    )
    return raw, profile_source(raw, filename="small.csv")


@pytest.fixture(scope="module")
def superstore():
    raw = load_source(SUPERSTORE.path)
    return raw, profile_source(raw, filename=SUPERSTORE.path.name)


def test_small_frame_has_every_role(small):
    _, profile = small
    roles = {c.name: c.role for c in profile.columns}
    assert roles == {
        "order_id": ColumnRole.IDENTIFIER, "date": ColumnRole.DATETIME, "amount": ColumnRole.NUMERIC,
        "segment": ColumnRole.CATEGORICAL, "flag": ColumnRole.BOOLEAN, "country": ColumnRole.CONSTANT,
        "note": ColumnRole.TEXT, "blank": ColumnRole.EMPTY,
    }


@pytest.mark.parametrize(
    ("column", "trace_types"),
    [
        ("amount", ["histogram", "box"]),
        ("segment", ["bar"]),
        ("flag", ["bar"]),
        ("country", ["bar"]),
        ("date", ["scatter"]),
    ],
)
def test_column_chart_traces_by_role(small, column, trace_types):
    raw, profile = small
    fig = column_chart(profile.column(column), raw[column])
    assert isinstance(fig, go.Figure)
    assert [t.type for t in fig.data] == trace_types


@pytest.mark.parametrize("column", ["order_id", "note", "blank"])
def test_column_chart_is_none_for_identifier_text_and_empty(small, column):
    raw, profile = small
    assert column_chart(profile.column(column), raw[column]) is None


def test_numeric_chart_reports_outliers(small):
    raw, profile = small
    col = profile.column("amount")
    fig = numeric_distribution_chart(col, raw["amount"])
    assert col.n_outliers >= 1
    assert f"{col.n_outliers:,} outliers" in fig.layout.title.text
    assert len(fig.data[0].x) == len(raw)


def test_top_values_chart_uses_profile_without_series(small):
    _, profile = small
    col = profile.column("segment")
    fig = top_values_chart(col)
    assert list(fig.data[0].y) == [v.value for v in col.top_values]
    assert sum(fig.data[0].x) == len(small[0])


def test_records_per_month_fills_gaps():
    s = pd.Series(["2024-01-05", "2024-01-20", "2024-04-02", None], name="d")
    profile = profile_source(s.to_frame(), filename="d.csv")
    fig = records_per_month_chart(profile.column("d"), s)
    assert list(fig.data[0].y) == [2, 0, 0, 1]


def test_to_datetime_handles_us_dates_and_missing():
    parsed = to_datetime(pd.Series(["11/8/2016", None, "1/2/2017"]))
    assert parsed.iloc[0] == pd.Timestamp("2016-11-08")
    assert pd.isna(parsed.iloc[1])


def test_role_breakdown_counts_all_columns(small, superstore):
    for _, profile in (small, superstore):
        fig = role_breakdown_chart(profile)
        assert sum(fig.data[0].x) == profile.n_cols
    counts = dict(zip(fig.data[0].y, fig.data[0].x))
    assert counts == {"identifier": 5, "datetime": 2, "numeric": 4, "categorical": 8, "text": 1, "constant": 1}


def test_missing_values_chart(small, superstore):
    _, profile = small
    fig = missing_values_chart(profile)
    assert list(fig.data[0].y) == ["blank", "note"]
    assert fig.data[0].x[0] == 1.0
    assert len(missing_values_chart(superstore[1]).data) == 0


def test_superstore_column_charts(superstore):
    raw, profile = superstore
    expected = {
        "Sales": ["histogram", "box"], "Region": ["bar"], "Country": ["bar"], "Order Date": ["scatter"],
        "Order ID": None, "Product Name": None,
    }
    for name, traces in expected.items():
        fig = column_chart(profile.column(name), raw[name])
        assert (fig and [t.type for t in fig.data]) == traces, name
    months = column_chart(profile.column("Order Date"), raw["Order Date"])
    assert len(months.data[0].x) == 48
    assert sum(months.data[0].y) == profile.n_rows
