from datetime import datetime

import pytest

from analyst.contracts import ColumnRole
from analyst.data.ingest import load_dataset
from analyst.data.profile import profile_dataset
from analyst.datasets import SUPERSTORE

pytestmark = pytest.mark.skipif(not SUPERSTORE.path.exists(), reason="Superstore data not downloaded")


@pytest.fixture(scope="module")
def profile():
    return profile_dataset(load_dataset(SUPERSTORE.path), name="superstore")


def test_shape_and_dates(profile):
    assert (profile.n_rows, profile.n_cols, profile.n_duplicate_rows) == (9994, 21, 0)
    assert profile.start == datetime(2014, 1, 3) and profile.end == datetime(2017, 12, 30)


def test_column_roles(profile):
    assert profile.columns_with_role(ColumnRole.DATETIME) == ["Order Date", "Ship Date"]
    assert profile.columns_with_role(ColumnRole.NUMERIC) == ["Sales", "Quantity", "Discount", "Profit"]
    assert profile.columns_with_role(ColumnRole.CONSTANT) == ["Country"]
    assert {"Row ID", "Order ID", "Customer ID", "Product ID", "Postal Code"} <= set(
        profile.columns_with_role(ColumnRole.IDENTIFIER)
    )
    assert {"Segment", "Region", "Category", "Sub-Category", "Ship Mode"} <= set(
        profile.columns_with_role(ColumnRole.CATEGORICAL)
    )


def test_known_statistics(profile):
    profit = profile.column("Profit")
    assert profit.n_negative == 1871
    assert profit.min == pytest.approx(-6599.978)
    assert profile.column("Sales").max == pytest.approx(22638.48)
    assert profile.column("Category").n_unique == 3


def test_known_quality_issues(profile):
    found = {(i.check, i.column) for i in profile.quality.issues}
    assert ("near_duplicate_rows", None) in found
    assert ("lost_leading_zeros", "Postal Code") in found
    assert ("constant_column", "Country") in found
    assert not any(i.check == "missing_values" for i in profile.quality.issues)
    assert profile.quality.passed
