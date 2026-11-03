"""Pure table/formatting helpers for the dashboard (no Streamlit calls)."""

from __future__ import annotations

import pandas as pd

from analyst.contracts import ColumnProfile, ColumnRole, DatasetProfile, QualityReport, Severity

SEVERITY_ICON = {Severity.ERROR: "⛔", Severity.WARNING: "⚠️", Severity.INFO: "ℹ️"}
SEVERITY_RANK = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
QUALITY_BANDS = ((85.0, "good", "green"), (60.0, "fair", "orange"), (0.0, "poor", "red"))
"""(minimum score, label, Streamlit colour) from best to worst."""


def quality_band(score: float) -> tuple[str, str]:
    """Label and colour name for a 0-100 quality score."""
    for threshold, label, color in QUALITY_BANDS:
        if score >= threshold:
            return label, color
    return QUALITY_BANDS[-1][1:]


def quality_issues_frame(report: QualityReport) -> pd.DataFrame:
    """Issues sorted by severity (errors first), then by number of affected values."""
    issues = sorted(report.issues, key=lambda i: (SEVERITY_RANK[i.severity], -i.n_affected))
    return pd.DataFrame(
        {
            "severity": [f"{SEVERITY_ICON[i.severity]} {i.severity.value}" for i in issues],
            "check": [i.check for i in issues],
            "column": [i.column or "" for i in issues],
            "affected": [i.n_affected for i in issues],
            "message": [i.message for i in issues],
        }
    )


def column_overview_frame(profile: DatasetProfile) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": [c.name for c in profile.columns],
            "role": [c.role.value if c.role else "" for c in profile.columns],
            "dtype": [c.dtype for c in profile.columns],
            "% missing": [100 * c.pct_missing for c in profile.columns],
            "unique": [c.n_unique for c in profile.columns],
            "key stats": [column_summary(c) for c in profile.columns],
        }
    )


def column_summary(c: ColumnProfile) -> str:
    """One-line, role-appropriate summary of a column profile."""
    if c.role is ColumnRole.NUMERIC and c.mean is not None:
        text = f"mean {c.mean:,.2f}, median {c.median:,.2f}, range [{c.min:,.2f}, {c.max:,.2f}]"
        return text + (f", {c.n_outliers:,} outliers" if c.n_outliers else "")
    if c.role is ColumnRole.DATETIME and c.min_date and c.max_date:
        return f"{c.min_date:%Y-%m-%d} to {c.max_date:%Y-%m-%d}"
    if c.role is ColumnRole.IDENTIFIER:
        return f"{c.pct_unique:.0%} unique" if c.pct_unique is not None else ""
    if c.top_values:
        return ", ".join(f"{v.value[:25]} ({v.count:,})" for v in c.top_values[:3])
    return ""


def sample_values(s: pd.Series, n: int = 10) -> pd.DataFrame:
    """First ``n`` distinct non-null values of ``s`` as a one-column frame."""
    values = s.dropna().drop_duplicates().head(n)
    return pd.DataFrame({str(s.name): values.astype(str).to_list()})
