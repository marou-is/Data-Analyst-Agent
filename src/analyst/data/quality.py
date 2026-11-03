from __future__ import annotations

import re

import pandas as pd

from analyst.contracts import ColumnProfile, ColumnRole, DatasetSpec, QualityIssue, QualityReport, Severity

_PENALTY = {Severity.INFO: 0.0, Severity.WARNING: 10.0, Severity.ERROR: 30.0}
MISSING_WARNING_PCT = 0.05
MISSING_ERROR_PCT = 0.5
NUMERIC_TEXT_PCT = 0.95
_CODE_NAME = re.compile(r"zip|postal|code|phone|sku", re.IGNORECASE)


def check_table_quality(
    raw: pd.DataFrame, columns: list[ColumnProfile], n_duplicate_rows: int = 0
) -> QualityReport:
    """Generic data-quality checks for any table, using its column profiles."""
    if len(raw) == 0:
        return _report([QualityIssue(check="empty", severity=Severity.ERROR, message="Dataset has no rows")])

    issues: list[QualityIssue] = []
    for col in columns:
        issues += _column_issues(raw[col.name], col)

    if n_duplicate_rows:
        issues.append(QualityIssue(
            check="duplicate_rows", severity=Severity.WARNING, n_affected=n_duplicate_rows,
            message=f"{n_duplicate_rows} fully duplicated rows",
        ))

    keys = [c.name for c in columns if c.role is ColumnRole.IDENTIFIER and c.pct_unique == 1 and c.n_missing == 0]
    rest = [c for c in raw.columns if c not in keys]
    if keys and rest:
        n_near = int(raw.duplicated(subset=rest).sum()) - n_duplicate_rows
        if n_near > 0:
            issues.append(QualityIssue(
                check="near_duplicate_rows", severity=Severity.WARNING, n_affected=n_near,
                message=f"{n_near} rows are identical to another row except for the unique key {keys}",
            ))
    return _report(issues)


def _column_issues(s: pd.Series, col: ColumnProfile) -> list[QualityIssue]:
    issues = []
    name = col.name
    if col.role is ColumnRole.EMPTY:
        return [QualityIssue(check="empty_column", severity=Severity.WARNING, column=name,
                             n_affected=col.n_missing, message=f"{name!r} has no values")]

    if col.n_missing:
        severity = (
            Severity.ERROR if col.pct_missing > MISSING_ERROR_PCT
            else Severity.WARNING if col.pct_missing > MISSING_WARNING_PCT
            else Severity.INFO
        )
        issues.append(QualityIssue(
            check="missing_values", severity=severity, column=name, n_affected=col.n_missing,
            message=f"{col.n_missing} missing values ({col.pct_missing:.1%}) in {name!r}",
        ))

    if col.role is ColumnRole.CONSTANT:
        value = col.top_values[0].value if col.top_values else s.dropna().iloc[0]
        issues.append(QualityIssue(
            check="constant_column", severity=Severity.INFO, column=name,
            message=f"{name!r} has a single value ({value!r}) and carries no information",
        ))

    non_null = s.dropna()
    if (
        col.role is ColumnRole.IDENTIFIER and pd.api.types.is_integer_dtype(s) and len(non_null)
        and _CODE_NAME.search(name)
    ):
        digits = non_null.abs().astype("int64").astype(str).str.len()
        n_short = int((digits < digits.max()).sum())
        if n_short and digits.max() <= 10 and digits.nunique() > 1:
            issues.append(QualityIssue(
                check="lost_leading_zeros", severity=Severity.WARNING, column=name, n_affected=n_short,
                message=(f"{n_short} values of code column {name!r} have fewer than {digits.max()} digits; "
                         "leading zeros were probably lost (store it as text)"),
            ))

    if col.role in (ColumnRole.CATEGORICAL, ColumnRole.TEXT) and len(non_null):
        as_number = pd.to_numeric(non_null.astype(str).str.replace(r"[$€£%,\s]", "", regex=True), errors="coerce")
        if as_number.notna().mean() >= NUMERIC_TEXT_PCT:
            issues.append(QualityIssue(
                check="numeric_stored_as_text", severity=Severity.WARNING, column=name, n_affected=len(non_null),
                message=f"{name!r} looks numeric but is stored as text",
            ))

    if col.role is ColumnRole.CATEGORICAL and len(non_null):
        normalized = non_null.astype(str).str.strip().str.lower().nunique()
        if normalized < col.n_unique:
            issues.append(QualityIssue(
                check="inconsistent_categories", severity=Severity.WARNING, column=name,
                n_affected=col.n_unique - normalized,
                message=f"{name!r} has {col.n_unique - normalized} categories differing only by case or whitespace",
            ))

    if col.role is ColumnRole.NUMERIC and col.n_outliers:
        pct = col.n_outliers / len(non_null)
        issues.append(QualityIssue(
            check="outliers", severity=Severity.INFO, column=name, n_affected=col.n_outliers,
            message=f"{col.n_outliers} values ({pct:.1%}) in {name!r} lie outside 1.5 x IQR",
        ))
    return issues


def check_quality(raw: pd.DataFrame, spec: DatasetSpec) -> QualityReport:
    """Run data-quality checks on a raw frame."""
    issues: list[QualityIssue] = []
    n = len(raw)
    if n == 0:
        issues.append(QualityIssue(check="empty", severity=Severity.ERROR, message="Dataset has no rows"))
        return _report(issues)

    for col in [spec.timestamp_col, *spec.target_cols]:
        n_missing = int(raw[col].isna().sum())
        if n_missing:
            pct = n_missing / n
            severity = Severity.ERROR if pct > 0.2 else Severity.WARNING
            issues.append(QualityIssue(
                check="missing_values", severity=severity, column=col, n_affected=n_missing,
                message=f"{n_missing} missing values ({pct:.1%}) in {col!r}",
            ))

    dupes = int(raw[spec.timestamp_col].duplicated().sum())
    if dupes:
        issues.append(QualityIssue(
            check="duplicate_timestamps", severity=Severity.WARNING, column=spec.timestamp_col,
            n_affected=dupes, message=f"{dupes} duplicated timestamps",
        ))
    return _report(issues)


def _report(issues: list[QualityIssue]) -> QualityReport:
    score = max(0.0, 100.0 - sum(_PENALTY[i.severity] for i in issues))
    return QualityReport(issues=issues, score=score)
