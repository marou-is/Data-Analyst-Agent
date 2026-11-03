import io

import pandas as pd
import pytest

from analyst.agent import profile_source
from analyst.contracts import ColumnProfile, ColumnRole, QualityIssue, QualityReport, Severity
from analyst.dashboard import state
from analyst.dashboard.tables import (
    column_overview_frame,
    column_summary,
    quality_band,
    quality_issues_frame,
    sample_values,
)


@pytest.mark.parametrize(("score", "label"), [(100, "good"), (85, "good"), (80, "fair"), (60, "fair"), (12, "poor")])
def test_quality_band(score, label):
    assert quality_band(score)[0] == label


def test_quality_issues_sorted_by_severity_then_affected():
    report = QualityReport(
        score=50,
        issues=[
            QualityIssue(check="a", severity=Severity.INFO, message="m", n_affected=99),
            QualityIssue(check="b", severity=Severity.ERROR, message="m", column="x"),
            QualityIssue(check="c", severity=Severity.WARNING, message="m", n_affected=1),
            QualityIssue(check="d", severity=Severity.WARNING, message="m", n_affected=5),
        ],
    )
    frame = quality_issues_frame(report)
    assert frame["check"].tolist() == ["b", "d", "c", "a"]
    assert frame["severity"].str.endswith(("error", "warning", "warning", "info")).all()
    assert frame["column"].tolist() == ["x", "", "", ""]


def test_column_overview_and_summary():
    profile = profile_source(pd.DataFrame({"v": [1.0, 2.0, None, 4.0], "c": ["a", "a", "b", "c"]}), filename="t.csv")
    frame = column_overview_frame(profile)
    assert frame.columns.tolist() == ["column", "role", "dtype", "% missing", "unique", "key stats"]
    assert frame.loc[0, "% missing"] == pytest.approx(25.0)
    assert frame.loc[0, "key stats"].startswith("mean 2.33")
    assert frame.loc[1, "key stats"].startswith("a (2)")
    ident = ColumnProfile(name="id", dtype="str", n_missing=0, pct_missing=0, n_unique=4, pct_unique=1.0,
                          role=ColumnRole.IDENTIFIER)
    assert column_summary(ident) == "100% unique"


def test_sample_values_are_distinct():
    assert sample_values(pd.Series(["a", "a", None, "b"], name="s"))["s"].tolist() == ["a", "b"]


def test_excel_sheets_and_load_table_with_sheet():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame({"x": [1, 2]}).to_excel(writer, sheet_name="first", index=False)
        pd.DataFrame({"y": [1.5, 2.5, 3.5]}).to_excel(writer, sheet_name="second", index=False)
    data = buf.getvalue()
    assert state.is_excel("book.XLSX") and not state.is_excel("book.csv")
    assert state.excel_sheets(data) == ["first", "second"]
    raw, profile = state.load_table(data, "book.xlsx", "second")
    assert raw.columns.tolist() == ["y"]
    assert profile.n_rows == 3 and profile.name == "book"
