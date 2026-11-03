import numpy as np
import pandas as pd

from analyst.contracts import Severity
from analyst.data.profile import profile_dataset


def issues(raw):
    return {i.check: i for i in profile_dataset(raw).quality.issues}


def test_missing_value_severity_thresholds():
    raw = pd.DataFrame({
        "few": [1.0] * 99 + [np.nan],
        "some": [1.0, 2.0] * 45 + [np.nan] * 10,
        "most": [1.0, 2.0] * 20 + [np.nan] * 60,
    })
    found = [i for i in profile_dataset(raw).quality.issues if i.check == "missing_values"]
    assert {i.column: i.severity for i in found} == {
        "few": Severity.INFO, "some": Severity.WARNING, "most": Severity.ERROR,
    }


def test_duplicates_and_near_duplicates():
    raw = pd.DataFrame({
        "Row ID": [1, 2, 3, 4],
        "Order": ["a", "a", "b", "b"],
        "Sales": [1.0, 1.0, 2.0, 3.0],
    })
    found = issues(raw)
    assert "duplicate_rows" not in found
    assert found["near_duplicate_rows"].n_affected == 1


def test_full_duplicates_are_not_double_counted():
    raw = pd.DataFrame({"Order": ["a", "a", "b"], "Sales": [1.0, 1.0, 2.0]})
    found = issues(raw)
    assert found["duplicate_rows"].n_affected == 1
    assert "near_duplicate_rows" not in found


def test_lost_leading_zeros_only_for_code_columns():
    raw = pd.DataFrame({"Row ID": range(1, 21), "Postal Code": [2108, 90036, 10001, 1040] * 5})
    found = [i for i in profile_dataset(raw).quality.issues if i.check == "lost_leading_zeros"]
    assert [(i.column, i.n_affected) for i in found] == [("Postal Code", 10)]


def test_numeric_stored_as_text_and_inconsistent_categories():
    raw = pd.DataFrame({
        "Price": ["$1,200", "$15", "$7.50", "$3"],
        "Segment": ["Consumer", "consumer ", "Corporate", "Consumer"],
    })
    found = issues(raw)
    assert found["numeric_stored_as_text"].column == "Price"
    assert found["inconsistent_categories"].column == "Segment"


def test_constant_column_and_score():
    raw = pd.DataFrame({"Country": ["US"] * 4, "Sales": [1.0, 2.0, 3.0, 4.0]})
    report = profile_dataset(raw).quality
    assert [i.check for i in report.issues] == ["constant_column"]
    assert report.score == 100 and report.passed


def test_clean_table_has_no_issues():
    raw = pd.DataFrame({"Region": ["West", "East", "South", "North"], "Sales": [1.0, 2.0, 3.0, 4.0]})
    assert profile_dataset(raw).quality.issues == []
