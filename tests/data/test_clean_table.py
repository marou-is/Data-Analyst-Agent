import numpy as np
import pandas as pd
import pytest

from analyst.data.clean import clean_table
from analyst.data.ingest import load_dataset
from analyst.data.profile import profile_dataset
from analyst.datasets import SUPERSTORE


def run(raw):
    return clean_table(raw, profile_dataset(raw))


def steps(result):
    return {s.name: s.rows_affected for s in result.steps}


@pytest.fixture
def raw():
    return pd.DataFrame({
        "Row ID": [1, 2, 3, 4, 5, 6],
        "Order Date": ["1/3/2014", "1/4/2014", "1/4/2014", "1/6/2014", "1/7/2014", "1/8/2014"],
        "Segment": ["Consumer", "consumer ", "consumer ", " Corporate", "Consumer", None],
        "Postal Code": [2108, 90036, 90036, 1040, 10001, 98103],
        "Sales": [1.0, 2.0, 2.0, np.nan, 5.0, 6.0],
    }, index=[10, 11, 12, 13, 14, 15])


def test_types_padding_and_categories(raw):
    result = run(raw)
    df = result.data
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])
    assert df.loc[10, "Order Date"] == pd.Timestamp("2014-01-03")
    assert df["Postal Code"].tolist() == ["02108", "90036", "01040", "10001", "98103"]
    assert df["Segment"].iloc[:4].tolist() == ["Consumer", "Consumer", "Corporate", "Consumer"]
    assert df["Segment"].isna().iloc[4]
    assert steps(result) == {
        "parse_datetime": 6, "pad_leading_zeros": 2, "strip_whitespace": 3,
        "unify_categories": 2, "drop_near_duplicate_rows": 1,
    }


def test_keeps_index_and_missing_values(raw):
    result = run(raw)
    assert result.data.index.tolist() == [10, 11, 13, 14, 15]
    assert (result.rows_before, result.rows_after) == (6, 5)
    assert np.isnan(result.data.loc[13, "Sales"])
    assert list(result.data.columns) == list(raw.columns)


def test_does_not_mutate_input(raw):
    before = raw.copy()
    run(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_full_duplicates_dropped_first_kept():
    raw = pd.DataFrame({"Region": ["West", "West", "East"], "Sales": [1.0, 1.0, 2.0]})
    result = run(raw)
    assert result.data.index.tolist() == [0, 2]
    assert steps(result) == {"drop_duplicate_rows": 1}


def test_clean_input_is_unchanged():
    raw = pd.DataFrame({"Region": ["West", "East", "South", "North"], "Sales": [1.0, 2.0, 3.0, 4.0]})
    result = run(raw)
    assert result.steps == []
    pd.testing.assert_frame_equal(result.data, raw)


@pytest.mark.skipif(not SUPERSTORE.path.exists(), reason="Superstore data not downloaded")
def test_superstore():
    raw = load_dataset(SUPERSTORE.path)
    result = run(raw)
    df = result.data
    assert (result.rows_before, result.rows_after, len(df)) == (9994, 9993, 9993)
    assert steps(result)["drop_near_duplicate_rows"] == 1
    assert steps(result)["pad_leading_zeros"] == 449
    assert (df["Postal Code"].str.len() == 5).all()
    assert "01040" in set(df["Postal Code"])
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])
    assert pd.api.types.is_datetime64_any_dtype(df["Ship Date"])
    assert df["Order Date"].min() == pd.Timestamp("2014-01-03")
    assert df.index.isin(raw.index).all()
