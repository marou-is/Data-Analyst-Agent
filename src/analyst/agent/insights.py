"""Rule-based insights derived from a finished report."""

from __future__ import annotations

from analyst.contracts import AnalysisReport, Insight, Severity


def generate_insights(report: AnalysisReport) -> list[Insight]:
    insights = [_overview(report)]
    insights += _quality(report)
    insights += [_anomaly(r) for r in report.anomalies]
    insights += [_forecast(r) for r in report.forecasts]
    return insights


def _overview(report: AnalysisReport) -> Insight:
    p = report.profile
    detail = f"From {p.start:%Y-%m-%d} to {p.end:%Y-%m-%d}, frequency {p.inferred_freq or 'unknown'}."
    if report.aggregation is not None and report.table_profile is not None:
        detail = (
            f"Aggregated from {report.table_profile.n_rows:,} rows by "
            f"{report.aggregation.spec.timestamp_col!r}. " + detail
        )
    return Insight(
        source="profile",
        title=f"{p.n_rows:,} observations of {', '.join(report.spec.target_cols)}",
        detail=detail,
    )


def _quality(report: AnalysisReport) -> list[Insight]:
    q = report.quality
    out = [
        Insight(
            source="quality",
            severity=Severity.INFO if q.score >= 80 else Severity.WARNING,
            title=f"Data quality score {q.score:.0f}/100",
            detail=f"{len(q.issues)} issue(s) found; {len(report.cleaning.steps)} cleaning step(s) applied.",
        )
    ]
    out += [
        Insight(source="quality", severity=i.severity, title=i.check.replace("_", " ").capitalize(), detail=i.message)
        for i in q.issues
        if i.severity is Severity.ERROR
    ]
    return out


def _anomaly(result) -> Insight:
    top = result.frame.nlargest(3, "score").index
    detail = f"Most anomalous: {', '.join(str(t) for t in top)}."
    if result.metrics:
        m = result.metrics
        detail += (
            f" Against labels: precision {m['precision']:.2f}, recall {m['recall']:.2f}, "
            f"{m['event_recall']:.0%} of labelled events detected."
        )
    share = result.n_anomalies / len(result.frame) if len(result.frame) else 0.0
    return Insight(
        source="anomaly",
        severity=Severity.WARNING if result.n_anomalies else Severity.INFO,
        title=f"{result.n_anomalies} anomalies in {result.target_col!r} ({share:.1%})",
        detail=detail,
    )


def _forecast(result) -> Insight:
    model_mae = result.metrics.get("mae")
    base_mae = result.baseline_metrics.get("mae")
    if model_mae is None or not base_mae:
        return Insight(source="forecast", title=f"Forecast for {result.target_col!r}", detail="No backtest metrics.")
    gain = 1 - model_mae / base_mae
    beats = gain > 0
    return Insight(
        source="forecast",
        severity=Severity.INFO if beats else Severity.WARNING,
        title=f"{result.model} {'beats' if beats else 'does not beat'} the naive baseline on {result.target_col!r}",
        detail=(
            f"Backtest MAE {model_mae:,.2f} vs baseline {base_mae:,.2f} ({gain:+.0%}) over "
            f"{result.horizon} steps; MAPE {result.metrics.get('mape', float('nan')):.1f}%."
        ),
    )
