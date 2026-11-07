"""Plotly figure builders. Pure functions: contract objects / Series in, ``go.Figure`` out."""

from __future__ import annotations

import warnings

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyst.contracts import AnalystAnswer, AnomalyResult, ColumnProfile, ColumnRole, DatasetProfile, ForecastResult
from analyst.dashboard.theme import AMBER, BLUE, COLORWAY, MUTED, TEAL, style_figure

_ROLE_ORDER = list(ColumnRole)
_TOP_VALUE_ROLES = (ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN, ColumnRole.CONSTANT)
_AREA_FILL = "rgba(67,85,219,.09)"


def anomaly_chart(result: AnomalyResult) -> go.Figure:
    f = result.frame
    flagged = f[f["is_anomaly"]]
    fig = go.Figure()
    fig.add_scatter(x=f.index, y=f["value"], mode="lines", name=result.target_col, line={"color": BLUE, "width": 1.4})
    fig.add_scatter(
        x=flagged.index, y=flagged["value"], mode="markers", name="unusual",
        marker={"color": AMBER, "size": 7, "line": {"color": "white", "width": 1}},
    )
    fig.update_layout(title=f"Unusual periods in {result.target_col} ({result.method.replace('_', ' ')})", hovermode="x unified")
    return style_figure(fig, height=340)


def forecast_chart(result: ForecastResult) -> go.Figure:
    bt, fc = result.backtest, result.forecast
    fig = go.Figure()
    fig.add_scatter(x=bt.index, y=bt["y"], mode="lines", name="actual (backtest)", line={"color": "#9AA1B1"})
    fig.add_scatter(x=bt.index, y=bt["yhat"], mode="lines", name="backtest prediction", line={"dash": "dot", "color": TEAL})
    fig.add_scatter(x=fc.index, y=fc["yhat_upper"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip")
    fig.add_scatter(
        x=fc.index, y=fc["yhat_lower"], mode="lines", line={"width": 0}, fill="tonexty",
        fillcolor="rgba(67,85,219,.12)", name="95% interval",
    )
    fig.add_scatter(x=fc.index, y=fc["yhat"], mode="lines", name="forecast", line={"color": BLUE, "width": 2.2})
    fig.update_layout(title=f"Forecast of {result.target_col} ({result.model.replace('_', ' ')})", hovermode="x unified")
    return style_figure(fig, height=340)


def trend_area_chart(series: pd.Series, highlights: pd.Series | None = None, height: int = 290) -> go.Figure:
    """Area chart of a time series; ``highlights`` (same units) are drawn as amber points."""
    fig = go.Figure()
    fig.add_scatter(
        x=series.index, y=series.to_numpy(), mode="lines", name=str(series.name or "value"),
        line={"color": BLUE, "width": 2.4}, fill="tozeroy", fillcolor=_AREA_FILL,
    )
    if highlights is not None and len(highlights):
        fig.add_scatter(
            x=highlights.index, y=highlights.to_numpy(), mode="markers", name="contains unusual days",
            marker={"color": AMBER, "size": 9, "line": {"color": "white", "width": 1.5}},
        )
    fig.update_layout(hovermode="x unified", showlegend=False)
    return style_figure(fig, height=height, show_title=False)


def breakdown_donut_chart(values: pd.Series, height: int = 290) -> go.Figure:
    """Donut of ``values`` (index = labels), at most 5 slices plus "Other"."""
    values = values.sort_values(ascending=False)
    if len(values) > 6:
        values = pd.concat([values.head(5), pd.Series({"Other": values.iloc[5:].sum()})])
    fig = go.Figure(go.Pie(
        labels=[str(v) for v in values.index], values=values.to_numpy(), hole=0.72, sort=False,
        marker={"colors": COLORWAY[: len(values)], "line": {"color": "white", "width": 2}},
        textinfo="none", hovertemplate="%{label}: %{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(legend={"orientation": "h", "y": -0.08, "x": 0.5, "xanchor": "center"})
    return style_figure(fig, height=height, show_title=False)


def answer_chart(answer: AnalystAnswer) -> go.Figure | None:
    """Chart for an analyst answer: first table column is x, second is y."""
    if answer.chart is None or answer.table is None or answer.table.shape[1] < 2:
        return None
    x, y = answer.table.columns[:2]
    if answer.chart == "bar":
        frame = answer.table.head(20)
        fig = go.Figure(go.Bar(
            x=frame[y], y=frame[x].astype(str), orientation="h", marker={"color": BLUE},
            hovertemplate="%{y}: %{x:,.2f}<extra></extra>",
        ))
        fig.update_layout(yaxis={"autorange": "reversed", "type": "category"}, xaxis_title=str(y))
        return style_figure(fig, height=max(260, 34 * len(frame) + 80), show_title=False)
    fig = go.Figure(go.Scatter(
        x=answer.table[x], y=answer.table[y], mode="lines+markers",
        line={"color": BLUE, "width": 2.2}, marker={"size": 4}, fill="tozeroy", fillcolor=_AREA_FILL,
    ))
    fig.update_layout(yaxis_title=str(y), hovermode="x unified")
    return style_figure(fig, height=320, show_title=False)


def role_breakdown_chart(profile: DatasetProfile) -> go.Figure:
    """Horizontal bar of the number of columns per semantic role."""
    counts = pd.Series([c.role for c in profile.columns if c.role is not None]).value_counts()
    roles = [r for r in _ROLE_ORDER if r in counts.index]
    fig = go.Figure(go.Bar(
        x=[int(counts[r]) for r in roles], y=[r.value for r in roles], orientation="h",
        text=[int(counts[r]) for r in roles], textposition="auto", marker={"color": BLUE},
    ))
    fig.update_layout(
        title="Columns by role", xaxis_title="columns", yaxis={"autorange": "reversed"},
        height=max(250, 45 * len(roles) + 120),
    )
    return style_figure(fig)


def missing_values_chart(profile: DatasetProfile) -> go.Figure:
    """Share of missing values for the columns that have any (no traces if none do)."""
    cols = sorted((c for c in profile.columns if c.n_missing), key=lambda c: c.pct_missing, reverse=True)
    fig = go.Figure()
    if cols:
        fig.add_bar(
            x=[c.pct_missing for c in cols], y=[c.name for c in cols], orientation="h",
            customdata=[c.n_missing for c in cols], marker={"color": AMBER},
            hovertemplate="%{y}: %{x:.1%} (%{customdata:,} missing)<extra></extra>",
        )
    fig.update_layout(
        title="Missing values by column", xaxis={"tickformat": ".0%", "range": [0, 1]},
        yaxis={"autorange": "reversed"}, height=max(250, 30 * len(cols) + 120),
    )
    return style_figure(fig)


def numeric_distribution_chart(col: ColumnProfile, series: pd.Series) -> go.Figure:
    """Histogram with a box plot (outliers shown as points) on a shared x axis."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Histogram(x=x, name="count", nbinsx=50, marker={"color": BLUE}), row=1, col=1)
    fig.add_trace(go.Box(x=x, name="", boxpoints="outliers", marker={"size": 3, "color": AMBER},
                         line={"color": MUTED}), row=2, col=1)
    outliers = f" · {col.n_outliers:,} outliers (1.5 × IQR)" if col.n_outliers is not None else ""
    fig.update_layout(title=f"Distribution of {col.name}{outliers}", showlegend=False, bargap=0.02)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    return style_figure(fig, height=380)


def top_values_chart(col: ColumnProfile, series: pd.Series | None = None, k: int = 15) -> go.Figure:
    """Bar of the ``k`` most frequent values; uses ``col.top_values`` when no series is given."""
    if series is not None:
        counts = series.dropna().astype(str).value_counts().head(k)
        values, freqs = counts.index.to_list(), counts.to_list()
    else:
        values, freqs = [v.value for v in col.top_values], [v.count for v in col.top_values]
    fig = go.Figure(go.Bar(x=freqs, y=values, orientation="h", marker={"color": BLUE}))
    shown = f"top {len(values)} of {col.n_unique:,}" if col.n_unique > len(values) else f"all {col.n_unique:,}"
    fig.update_layout(
        title=f"Most frequent values of {col.name} ({shown})", xaxis_title="rows",
        yaxis={"autorange": "reversed", "type": "category"}, height=max(250, 28 * len(values) + 120),
    )
    return style_figure(fig)


def records_per_month_chart(col: ColumnProfile, series: pd.Series) -> go.Figure:
    """Line of row counts per calendar month (months without rows count as 0)."""
    months = to_datetime(series).dropna().dt.to_period("M")
    counts = months.value_counts().sort_index()
    if len(counts):
        counts = counts.reindex(pd.period_range(counts.index.min(), counts.index.max(), freq="M"), fill_value=0)
    fig = go.Figure(go.Scatter(
        x=counts.index.to_timestamp(), y=counts.to_numpy(), mode="lines+markers",
        line={"color": BLUE, "width": 2}, marker={"size": 4},
    ))
    fig.update_layout(title=f"Records per month by {col.name}", yaxis_title="rows", hovermode="x unified")
    return style_figure(fig, height=340)


def column_chart(col: ColumnProfile, series: pd.Series) -> go.Figure | None:
    """Role-appropriate chart for one column, or ``None`` (identifier, text, empty)."""
    if col.role is ColumnRole.NUMERIC:
        return numeric_distribution_chart(col, series)
    if col.role in _TOP_VALUE_ROLES:
        return top_values_chart(col, series)
    if col.role is ColumnRole.DATETIME:
        return records_per_month_chart(col, series)
    return None


def to_datetime(series: pd.Series) -> pd.Series:
    """Parse a raw column as datetimes; unparseable values become ``NaT``."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    text = series.dropna().astype(str)
    fmt = pd.tseries.api.guess_datetime_format(text.iloc[0]) if len(text) else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series.astype(str).where(series.notna()), errors="coerce", format=fmt or "mixed")
