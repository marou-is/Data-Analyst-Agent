"""Aggregate a transactional table (one row per event) into a regular time series."""

from __future__ import annotations

import re

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset, Tick

from analyst.contracts import (
    GROUP, TIMESTAMP, AggregateResult, AggregationSpec, ColumnProfile, ColumnRole, DatasetProfile, MetricSpec,
)
from analyst.data.profile import _parse_dates
from analyst.data.quality import _CODE_NAME

ZERO_FILLED_AGGS = ("sum", "count", "nunique")
MISSING_GROUP = "(missing)"
_PERIOD = "__period"


def default_aggregation(profile: DatasetProfile, freq: str = "D") -> AggregationSpec:
    """Aggregation spec for a table profile: bucket by its first datetime-role column.

    Metrics, in column order:

    * every numeric-role column: ``sum``, or ``mean`` for rate-like columns whose
      values all lie in [0, 1] (e.g. a discount rate);
    * ``n_rows``: the number of rows per period (``count``);
    * ``nunique`` of each non-unique identifier column that names an entity
      (e.g. ``Order ID`` -> number of orders), excluding code-like columns such
      as zip/postal codes.

    Metric names are the snake_case source column name (``"Sales"`` -> ``sales``,
    ``"Unit Price"`` -> ``unit_price``), ``n_<snake_case>`` for nunique metrics
    (``n_order_id``) and ``n_rows`` for the row count; clashes get a ``_2``, ``_3`` suffix.
    """
    timestamps = profile.columns_with_role(ColumnRole.DATETIME)
    if not timestamps:
        raise ValueError("no datetime column to aggregate by")
    metrics: list[tuple[str, str, str | None]] = [
        (snake_case(c.name), "mean" if _is_rate(c) else "sum", c.name)
        for c in profile.columns if c.role is ColumnRole.NUMERIC
    ]
    metrics.append(("n_rows", "count", None))
    metrics += [
        (f"n_{snake_case(c.name)}", "nunique", c.name)
        for c in profile.columns
        if c.role is ColumnRole.IDENTIFIER and c.pct_unique is not None and c.pct_unique < 1
        and not _CODE_NAME.search(c.name)
    ]
    used: set[str] = set()
    specs = []
    for name, agg, column in metrics:
        unique, i = name, 1
        while unique in used:
            i += 1
            unique = f"{name}_{i}"
        used.add(unique)
        specs.append(MetricSpec(name=unique, agg=agg, column=column))
    return AggregationSpec(timestamp_col=timestamps[0], freq=freq, metrics=specs)


def aggregate(table: pd.DataFrame, spec: AggregationSpec) -> AggregateResult:
    """Aggregate ``table`` per ``spec.freq`` period (see ``AggregateResult``).

    Each row goes to the latest period start at or before its timestamp
    (``"D"``: midnight, ``"W-MON"``: Monday, ``"W"``: Sunday, ``"MS"``: first of month).
    Rows with a missing or unparseable timestamp are ignored; missing group values
    become the group ``"(missing)"``. Every period from the first to the last is
    present (for every group, on the same shared range): empty periods are 0 for
    sum/count/nunique and NaN for mean/median/min/max.
    """
    needed = [spec.timestamp_col, *([spec.group_by] if spec.group_by else []), *(m.column for m in spec.metrics if m.column)]
    missing = sorted({c for c in needed if c not in table.columns})
    if missing:
        raise ValueError(f"table is missing columns {missing}")

    names = [m.name for m in spec.metrics]
    ts = _to_datetime(table[spec.timestamp_col]).dropna()
    if ts.empty:
        return _empty_result(spec, names)

    offset = to_offset(spec.freq)
    periods = pd.date_range(_period_start(ts.min(), offset), ts.max(), freq=offset)
    keys = [pd.Series(periods[periods.searchsorted(ts, side="right") - 1], index=ts.index, name=_PERIOD)]
    groups: list[str] = []
    if spec.group_by:
        values = table.loc[ts.index, spec.group_by]
        keys.append(values.astype(str).where(values.notna(), MISSING_GROUP).rename(GROUP))
        groups = sorted(keys[1].unique())

    rows = table.loc[ts.index]
    columns = {}
    for m in spec.metrics:
        if m.agg == "count":
            columns[m.name] = keys[0].groupby(keys).size()
        else:
            values = rows[m.column] if m.agg == "nunique" else _to_numeric(rows[m.column])
            columns[m.name] = values.groupby(keys).agg(m.agg)
    data = pd.DataFrame(columns)

    if spec.group_by:
        data = data.reindex(pd.MultiIndex.from_product([periods, groups], names=[_PERIOD, GROUP]))
        data = data.reset_index(level=GROUP)[[GROUP, *names]]
    else:
        data = data.reindex(periods)
    zero_filled = [m.name for m in spec.metrics if m.agg in ZERO_FILLED_AGGS]
    if zero_filled:
        data[zero_filled] = data[zero_filled].fillna(0.0)
    data[names] = data[names].astype("float64")
    data.index = pd.DatetimeIndex(data.index, name=TIMESTAMP)
    return AggregateResult(spec=spec, data=data, groups=groups)


def snake_case(name: str) -> str:
    """``"Sub-Category"`` -> ``sub_category``, ``"unitPrice"`` -> ``unit_price``."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower() or "column"


def _is_rate(col: ColumnProfile) -> bool:
    return col.min is not None and col.max is not None and col.min >= 0 and col.max <= 1


def _to_datetime(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = _parse_dates(s, force=True)
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s


def _to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce")


def _period_start(t: pd.Timestamp, offset: BaseOffset) -> pd.Timestamp:
    if isinstance(offset, Tick):
        return t.floor(offset)
    return offset.rollback(t.normalize())


def _empty_result(spec: AggregationSpec, names: list[str]) -> AggregateResult:
    data = pd.DataFrame({n: pd.Series(dtype="float64") for n in names}, index=pd.DatetimeIndex([], name=TIMESTAMP))
    if spec.group_by:
        data.insert(0, GROUP, pd.Series(dtype=str))
    return AggregateResult(spec=spec, data=data, groups=[])
