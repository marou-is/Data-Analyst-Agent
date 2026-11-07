"""Rule-based analyst: answers common business questions with evidence.

Stand-in for the LLM agent (Gemini + LangGraph, milestone 4), which must return the
same ``AnalystAnswer`` so the dashboard does not change.
"""

from __future__ import annotations

import re

import pandas as pd

from analyst.agent.pipeline import with_margin_ratio
from analyst.contracts import (
    AnalysisConfig, AnalysisReport, AnalystAnswer, ColumnRole, DatasetProfile, QualityIssue, Severity,
)
from analyst.data.clean import clean_table
from analyst.ml.row_anomaly import detect_row_anomalies

ENGINE = "Rule-based analyst (LLM not connected)"
SUGGESTED_QUESTIONS = (
    "Which category generates the most revenue?",
    "How has revenue changed over time?",
    "Are there unusual values in my dataset?",
    "What are the main data quality issues?",
)
_INTENTS = (
    ("quality", r"quality|missing|duplicat|clean|issue|problem|error"),
    ("anomalies", r"unusual|anomal|outlier|strange|suspicious|weird"),
    ("forecast", r"forecast|predict|next|future|expect"),
    ("trend", r"trend|over time|chang|grow|evol|declin|increas|decreas|month|year|week|season"),
    ("breakdown", r"which|top|most|best|worst|highest|lowest|least|\bby\b|\bper\b|breakdown|compar|share|split"),
)
_REVENUE = re.compile(r"sales|revenue|turnover|amount", re.IGNORECASE)
_PROFIT = re.compile(r"profit|margin|income|earning", re.IGNORECASE)
_PREFERRED_DIMENSIONS = ("category", "segment", "region", "type", "channel", "product")
_ASCENDING = re.compile(r"worst|lowest|least|bottom|smallest|weakest")
MAX_DEFAULT_DIMENSION_VALUES = 12
_SEVERITY_RANK = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
_PERIOD_WORDS = (("min", "minute"), ("h", "hour"), ("d", "day"), ("w", "week"), ("m", "month"),
                 ("q", "quarter"), ("y", "year"))


def prepare_table(raw: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    """The clean table answers are computed on (dates parsed, duplicates removed)."""
    return clean_table(raw, profile).data


def answer_question(
    question: str, table: pd.DataFrame, profile: DatasetProfile, report: AnalysisReport | None = None
) -> AnalystAnswer:
    """Answer ``question`` about ``table`` (a clean table, see ``prepare_table``)."""
    intent = classify(question)
    handler = {
        "breakdown": _breakdown, "trend": _trend, "anomalies": _anomalies,
        "quality": _quality, "forecast": _forecast, "overview": _overview,
    }[intent]
    return handler(question, question.lower(), table, profile, report)


def classify(question: str) -> str:
    q = question.lower()
    for intent, pattern in _INTENTS:
        if re.search(pattern, q):
            return intent
    return "overview"


def find_measure(question: str, profile: DatasetProfile) -> str | None:
    """Numeric column named in the question, else revenue-like, else the first numeric."""
    q = question.lower()
    numeric = profile.columns_with_role(ColumnRole.NUMERIC)
    for col in sorted(numeric, key=len, reverse=True):
        if col.lower() in q:
            return col
    for pattern in (_PROFIT, _REVENUE):
        if pattern.search(q):
            match = next((c for c in numeric if pattern.search(c)), None)
            if match:
                return match
    return next((c for c in numeric if _REVENUE.search(c)), numeric[0] if numeric else None)


def find_dimension(question: str, profile: DatasetProfile) -> str | None:
    """Categorical column named in the question, else a small, business-like one."""
    q = question.lower()
    cats = [c for c in profile.columns if c.role in (ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN)]
    for col in sorted(cats, key=lambda c: len(c.name), reverse=True):
        if any(v in q for v in _name_variants(col.name)):
            return col.name
    small = [c.name for c in cats if c.n_unique <= MAX_DEFAULT_DIMENSION_VALUES]
    for word in _PREFERRED_DIMENSIONS:
        match = next((c for c in small if word in c.lower()), None)
        if match:
            return match
    return (small or [c.name for c in cats] or [None])[0]


def aggregated_metric(report: AnalysisReport | None, column: str | None) -> str | None:
    """Name of the time-series metric built from ``column`` in ``report``, if any."""
    if report is None or column is None:
        return None
    if report.aggregation is not None:
        return next(
            (m.name for m in report.aggregation.spec.metrics if m.column == column and m.agg != "nunique"), None
        )
    return column if column in report.spec.target_cols else None


def format_value(column: str, value: float) -> str:
    money = bool(_REVENUE.search(column) or _PROFIT.search(column))
    text = f"{abs(value):,.0f}" if abs(value) >= 100 else f"{abs(value):,.2f}"
    sign = "-" if value < 0 else ""
    return f"{sign}${text}" if money else f"{sign}{text}"


def period_word(freq: str | None, n: int = 2) -> str:
    """``("D", 3)`` -> ``"days"``, ``("30min", 3)`` -> ``"30min periods"``."""
    match = re.fullmatch(r"(\d*)([A-Za-z]+)(-\w+)?", freq or "")
    if not match:
        return "periods"
    multiple, unit = match.group(1), match.group(2).lower()
    if multiple not in ("", "1"):
        return f"{freq} periods"
    word = next((w for prefix, w in _PERIOD_WORDS if unit.startswith(prefix)), "period")
    return word if n == 1 else word + "s"


def _breakdown(question, q, table, profile, report) -> AnalystAnswer:
    measure, dim = find_measure(q, profile), find_dimension(q, profile)
    if measure is None or dim is None:
        return _overview(question, q, table, profile, report, note="I need a numeric and a categorical column for a breakdown.")
    agg = "mean" if _is_rate(profile, measure) else "sum"
    frame = (
        table.groupby(table[dim].astype(str), dropna=False)[measure].agg([agg, "count"])
        .rename(columns={agg: measure, "count": "rows"})
        .sort_values(measure, ascending=bool(_ASCENDING.search(q)))
        .reset_index()
    )
    total = float(table[measure].sum())
    if agg == "sum" and total:
        frame["share"] = frame[measure] / total
    profit = next((c for c in profile.columns_with_role(ColumnRole.NUMERIC) if _PROFIT.search(c)), None)
    if profit and profit != measure and _REVENUE.search(measure):
        sums = table.groupby(table[dim].astype(str))[[profit, measure]].sum()
        frame["margin"] = frame[dim].map(sums[profit] / sums[measure])

    word = "lowest" if _ASCENDING.search(q) else "highest"
    label = f"average {measure}" if agg == "mean" else f"total {measure}"
    lead = frame.iloc[0]
    summary = f"{lead[dim]} has the {word} {label}: {format_value(measure, lead[measure])}"
    if "share" in frame:
        summary += f" ({lead['share']:.0%} of {format_value(measure, total)})"
    if len(frame) > 1:
        runner = frame.iloc[1]
        summary += f", followed by {runner[dim]} ({format_value(measure, runner[measure])})"
    summary += "."
    if "margin" in frame:
        best = frame.loc[frame["margin"].idxmax()]
        summary += f" The best profit margin is {best[dim]} at {best['margin']:.1%}."

    evidence = []
    for _, row in frame.head(5).iterrows():
        line = f"{row[dim]}: {format_value(measure, row[measure])}"
        if "share" in frame:
            line += f" ({row['share']:.1%})"
        if "margin" in frame:
            line += f", margin {row['margin']:.1%}"
        evidence.append(line + f", {int(row['rows']):,} rows")
    evidence.append(f"{len(frame)} values of {dim!r} compared on {len(table):,} cleaned rows.")
    return AnalystAnswer(
        question=question, intent="breakdown", summary=summary, evidence=evidence,
        table=frame, chart="bar", metric=measure, engine=ENGINE,
    )


def _trend(question, q, table, profile, report) -> AnalystAnswer:
    measure = find_measure(q, profile)
    ts = next((c for c in profile.columns_with_role(ColumnRole.DATETIME)
               if c in table and pd.api.types.is_datetime64_any_dtype(table[c])), None)
    if measure is None or ts is None:
        return _overview(question, q, table, profile, report, note="I need a date column and a numeric column for a trend.")
    agg = "mean" if _is_rate(profile, measure) else "sum"
    series = table.set_index(ts)[measure].sort_index()
    monthly = series.resample("MS").agg(agg)
    yearly = series.groupby(series.index.year).agg(agg)
    months_per_year = series.groupby(series.index.year).apply(lambda s: s.index.month.nunique())

    if len(yearly) >= 2:
        (y0, v0), (y1, v1) = list(yearly.items())[-2:]
        change = (v1 - v0) / abs(v0) if v0 else float("nan")
        partial = " (partial year)" if months_per_year[y1] < 12 else ""
        direction = "grew" if v1 >= v0 else "fell"
        summary = (f"{measure} {direction} {change:+.1%} from {y0} ({format_value(measure, v0)}) "
                   f"to {y1}{partial} ({format_value(measure, v1)}).")
    else:
        summary = f"{measure} covers a single year: {format_value(measure, float(yearly.iloc[0]))}."
    best, worst = monthly.idxmax(), monthly.idxmin()
    summary += f" The strongest month was {best:%B %Y} ({format_value(measure, monthly[best])})."

    evidence = [f"{year}: {format_value(measure, value)}" for year, value in yearly.items()]
    evidence += [
        f"Weakest month: {worst:%B %Y} ({format_value(measure, monthly[worst])}).",
        f"Average month: {format_value(measure, float(monthly.mean()))} over {len(monthly)} months by {ts!r}.",
    ]
    frame = pd.DataFrame({"month": monthly.index, measure: monthly.to_numpy()})
    return AnalystAnswer(
        question=question, intent="trend", summary=summary, evidence=evidence,
        table=frame, chart="line", metric=measure, engine=ENGINE,
    )


def _anomalies(question, q, table, profile, report) -> AnalystAnswer:
    rows = report.row_anomalies if report is not None else None
    if rows is None:
        rows = detect_row_anomalies(table, profile, with_margin_ratio(AnalysisConfig(), profile))
    measure = find_measure(q, profile)
    flagged = rows.frame[rows.frame["is_anomaly"]].sort_values("score", ascending=False)
    keys = [c.name for c in profile.columns if c.role is ColumnRole.IDENTIFIER and c.pct_unique == 1][:1]
    context = keys + profile.columns_with_role(ColumnRole.DATETIME)[:1] + [find_dimension(q, profile)] + rows.features
    context = [c for c in dict.fromkeys(context) if c and c in table.columns]
    frame = table.loc[flagged.index, context].assign(reason=flagged["reason"]).head(15)

    n, total = rows.n_anomalies, len(rows.frame)
    summary = (f"{rows.method.replace('_', ' ').title()} flagged {n:,} of {total:,} rows ({n / max(total, 1):.1%}) "
               f"as unusual, based on {', '.join(rows.features)}.")
    if n:
        summary += f" The most unusual: {flagged['reason'].iloc[0]}."
    evidence = [f"Row {idx}: {reason}" for idx, reason in flagged["reason"].head(3).items()]
    metric = aggregated_metric(report, measure)
    if report is not None:
        for result in report.anomalies:
            if result.target_col == metric:
                evidence.append(
                    f"Over time: {result.n_anomalies} unusual {period_word(report.spec.freq, result.n_anomalies)} "
                    f"in {metric} ({result.method.replace('_', ' ')})."
                )
    return AnalystAnswer(
        question=question, intent="anomalies", summary=summary, evidence=evidence,
        table=frame, metric=measure, engine=ENGINE,
    )


def _quality(question, q, table, profile, report) -> AnalystAnswer:
    quality = profile.quality
    issues: list[QualityIssue] = sorted(quality.issues, key=lambda i: (_SEVERITY_RANK[i.severity], -i.n_affected)) if quality else []
    counts = {s: sum(i.severity is s for i in issues) for s in Severity}
    score = f"{quality.score:.0f}/100" if quality else "n/a"
    summary = (f"Data quality score {score}: {counts[Severity.ERROR]} errors, "
               f"{counts[Severity.WARNING]} warnings and {counts[Severity.INFO]} notes.")
    if issues:
        summary += f" Most important: {issues[0].message}."
    evidence = [i.message for i in issues[:8]]
    if report is not None:
        evidence += [f"Fixed during cleaning: {s.description}" for s in report.cleaning.steps if s.name != "aggregate"][:5]
    frame = pd.DataFrame({
        "severity": [i.severity.value for i in issues],
        "check": [i.check for i in issues],
        "column": [i.column or "" for i in issues],
        "affected": [i.n_affected for i in issues],
        "message": [i.message for i in issues],
    })
    return AnalystAnswer(question=question, intent="quality", summary=summary, evidence=evidence, table=frame, engine=ENGINE)


def _forecast(question, q, table, profile, report) -> AnalystAnswer:
    measure = find_measure(q, profile)
    metric = aggregated_metric(report, measure)
    result = next((f for f in report.forecasts if f.target_col == metric), None) if report else None
    if result is None:
        return _overview(question, q, table, profile, report, note=f"No forecast is available for {measure!r}.")
    fc = result.forecast
    mae, base = result.metrics.get("mae"), result.baseline_metrics.get("mae")
    summary = (f"Over the next {result.horizon} {period_word(report.spec.freq, result.horizon)}, {measure} is expected to total "
               f"{format_value(measure, float(fc['yhat'].sum()))} ({result.model.replace('_', ' ')}).")
    evidence = [f"Forecast from {fc.index.min():%Y-%m-%d} to {fc.index.max():%Y-%m-%d}."]
    if mae is not None and base:
        gain = 1 - mae / base
        verdict = "beats" if gain > 0 else "does not beat"
        evidence.append(f"Backtest MAE {mae:,.2f} vs naive baseline {base:,.2f}: the model {verdict} it ({gain:+.0%}).")
    frame = fc.reset_index(names="date")[["date", "yhat", "yhat_lower", "yhat_upper"]]
    return AnalystAnswer(
        question=question, intent="forecast", summary=summary, evidence=evidence,
        table=frame, chart="line", metric=measure, engine=ENGINE,
    )


def _overview(question, q, table, profile, report, note: str | None = None) -> AnalystAnswer:
    span = f" from {profile.start:%Y-%m-%d} to {profile.end:%Y-%m-%d}" if profile.start and profile.end else ""
    summary = (note + " " if note else "") + (
        f"The dataset has {profile.n_rows:,} rows and {profile.n_cols} columns{span}. "
        "I can answer questions about breakdowns, trends, unusual values, data quality and forecasts."
    )
    evidence = [i.title for i in report.insights[:5]] if report is not None else []
    evidence += [f"Try: {s}" for s in SUGGESTED_QUESTIONS]
    return AnalystAnswer(question=question, intent="overview", summary=summary, evidence=evidence, engine=ENGINE)


def _is_rate(profile: DatasetProfile, column: str) -> bool:
    col = profile.column(column)
    return col.min is not None and col.max is not None and col.min >= 0 and col.max <= 1


def _name_variants(name: str) -> set[str]:
    n = name.lower()
    variants = {n, n.replace("-", " "), n.replace("-", ""), n.replace("_", " ")}
    variants |= {v + "s" for v in list(variants)}
    if n.endswith("y"):
        variants.add(n[:-1] + "ies")
    return variants
