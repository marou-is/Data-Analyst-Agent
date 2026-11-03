from __future__ import annotations

import pandas as pd

from analyst.contracts import DatasetSpec, QualityIssue, QualityReport, Severity

_PENALTY = {Severity.INFO: 0.0, Severity.WARNING: 10.0, Severity.ERROR: 30.0}


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
