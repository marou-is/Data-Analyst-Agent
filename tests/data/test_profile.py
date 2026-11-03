from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from analyst.contracts import ColumnRole, DatasetProfile
from analyst.data.profile import format_profile, infer_role, profile_column, profile_dataset


@pytest.mark.parametrize(
    ("values", "name", "role"),
    [
        ([1, 2, 3, 4], "Row ID", ColumnRole.IDENTIFIER),
        (["A-1", "A-1", "B-2", "C-3"], "Order ID", ColumnRole.IDENTIFIER),
        (["a", "b", "c", "d"], "customerId", ColumnRole.IDENTIFIER),
        ([10, 20, 30, 40], "Paid", ColumnRole.IDENTIFIER),  # unique monotonic integers
        ([10.0, 5.0, 30.0, 40.0], "Paid", ColumnRole.NUMERIC),
        (["11/8/2016", "6/12/2016", "1/3/2014", None], "Order Date", ColumnRole.DATETIME),
        (["2024-01-01 10:00", "2024-01-02 11:30", "2024-02-01 00:00", "2024-03-01 09:15"], "when", ColumnRole.DATETIME),
        (["West", "East", "West", "South"], "Region", ColumnRole.CATEGORICAL),
        (["yes", "no", "yes", None], "Returned", ColumnRole.BOOLEAN),
        (["US", "US", "US", "US"], "Country", ColumnRole.CONSTANT),
        ([None, None, None, None], "Notes", ColumnRole.EMPTY),
        (["a long free-text product description number one", "another long product description entirely",
          "yet another long product description here", "a final long product description string"], "Product", ColumnRole.TEXT),
        (["2016", "2017", "2018", "2019"], "Fiscal", ColumnRole.CATEGORICAL),  # digits only: not parsed as dates
    ],
)
def test_infer_role(values, name, role):
    assert infer_role(pd.Series(values, name=name))[0] is role


def test_numeric_column_stats():
    col = profile_column(pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, -1.0, 0.0, np.nan], name="Profit"))
    assert col.role is ColumnRole.NUMERIC
    assert (col.n_missing, col.n_unique, col.n_negative, col.n_zeros) == (1, 7, 1, 1)
    assert col.min == -1.0 and col.max == 100.0 and col.median == 2.0
    assert col.n_outliers == 1


def test_datetime_column_range():
    col = profile_column(pd.Series(["11/8/2016", "6/12/2016", "1/3/2014"], name="Order Date"))
    assert col.min_date == datetime(2014, 1, 3) and col.max_date == datetime(2016, 11, 8)


def test_categorical_top_values():
    col = profile_column(pd.Series(["West"] * 3 + ["East"] * 2 + ["South"], name="Region"), top_k=2)
    assert [(v.value, v.count) for v in col.top_values] == [("West", 3), ("East", 2)]


def test_profile_dataset_is_json_round_trippable():
    raw = pd.DataFrame({
        "Order Date": ["1/1/2020", "1/2/2020", "1/2/2020"],
        "Region": ["West", "East", "East"],
        "Sales": [1.0, 2.0, 2.0],
    })
    p = profile_dataset(raw, name="tiny")
    assert (p.n_rows, p.n_cols, p.n_duplicate_rows) == (3, 3, 1)
    assert p.start == datetime(2020, 1, 1) and p.end == datetime(2020, 1, 2)
    assert p.columns_with_role(ColumnRole.NUMERIC) == ["Sales"]
    assert DatasetProfile.model_validate_json(p.model_dump_json()) == p
    assert "tiny" in format_profile(p)


def test_empty_frame():
    p = profile_dataset(pd.DataFrame({"a": []}))
    assert p.n_rows == 0 and not p.quality.passed
