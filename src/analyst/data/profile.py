"""Dataset profiling: column roles, statistics and data-quality issues.

CLI: ``python -m analyst.data.profile data/raw/superstore.csv [--json out.json]``
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd

from analyst.contracts import ColumnProfile, ColumnRole, DatasetProfile, DatasetSpec, ValueCount
from analyst.data.quality import check_table_quality

_ID_NAME = re.compile(r"(^|[\s_\-])(id|key|code|zip|postal|sku|uuid)([\s_\-]|$)", re.IGNORECASE)
_CAMEL_ID_NAME = re.compile(r"[a-z]I[dD]$")
_BOOL_VALUES = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"}
DATETIME_PARSE_THRESHOLD = 0.95
TEXT_MIN_MEAN_LENGTH = 30


def profile_dataset(raw: pd.DataFrame, name: str = "dataset", top_k: int = 5) -> DatasetProfile:
    """Profile any table: per-column role and statistics plus data-quality issues."""
    n_dup = int(raw.duplicated().sum()) if len(raw) else 0
    columns = [profile_column(raw[col], top_k=top_k) for col in raw.columns]
    datetime_cols = [c for c in columns if c.role is ColumnRole.DATETIME]
    primary = datetime_cols[0] if datetime_cols else None
    return DatasetProfile(
        name=name,
        n_rows=len(raw),
        n_cols=raw.shape[1],
        n_duplicate_rows=n_dup,
        memory_mb=round(raw.memory_usage(deep=True).sum() / 2**20, 3),
        start=primary.min_date if primary else None,
        end=primary.max_date if primary else None,
        columns=columns,
        quality=check_table_quality(raw, columns, n_duplicate_rows=n_dup),
    )


def profile_column(s: pd.Series, top_k: int = 5) -> ColumnProfile:
    n = len(s)
    non_null = s.dropna()
    n_unique = int(non_null.nunique())
    role, parsed_dates = infer_role(s)
    stats: dict = {}

    if role is ColumnRole.NUMERIC or (role is ColumnRole.IDENTIFIER and pd.api.types.is_numeric_dtype(s)):
        stats = _numeric_stats(non_null.astype("float64"), with_distribution=role is ColumnRole.NUMERIC)
    if role is ColumnRole.DATETIME and parsed_dates is not None and parsed_dates.notna().any():
        stats = {"min_date": parsed_dates.min().to_pydatetime(), "max_date": parsed_dates.max().to_pydatetime()}
    if role in (ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN, ColumnRole.CONSTANT, ColumnRole.TEXT):
        counts = non_null.astype(str).value_counts().head(top_k)
        stats["top_values"] = [ValueCount(value=str(v), count=int(c)) for v, c in counts.items()]

    return ColumnProfile(
        name=str(s.name),
        dtype=str(s.dtype),
        n_missing=int(n - len(non_null)),
        pct_missing=float((n - len(non_null)) / n) if n else 0.0,
        n_unique=n_unique,
        pct_unique=float(n_unique / len(non_null)) if len(non_null) else None,
        role=role,
        **stats,
    )


def infer_role(s: pd.Series) -> tuple[ColumnRole, pd.Series | None]:
    """Return the semantic role of ``s`` and, for datetime columns, the parsed values."""
    non_null = s.dropna()
    n_unique = non_null.nunique()
    if n_unique == 0:
        return ColumnRole.EMPTY, None
    if n_unique == 1:
        return ColumnRole.CONSTANT, None
    if pd.api.types.is_bool_dtype(s) or (
        n_unique == 2 and {str(v).strip().lower() for v in non_null.unique()} <= _BOOL_VALUES
    ):
        return ColumnRole.BOOLEAN, None
    if pd.api.types.is_datetime64_any_dtype(s):
        return ColumnRole.DATETIME, pd.to_datetime(s)

    name = str(s.name)
    is_numeric = pd.api.types.is_numeric_dtype(s)
    if not is_numeric:
        parsed = _parse_dates(non_null)
        if parsed is not None:
            return ColumnRole.DATETIME, _parse_dates(s, force=True)
    if _ID_NAME.search(name) or _CAMEL_ID_NAME.search(name):
        return ColumnRole.IDENTIFIER, None
    if is_numeric:
        if pd.api.types.is_integer_dtype(s) and n_unique == len(non_null) and non_null.is_monotonic_increasing:
            return ColumnRole.IDENTIFIER, None
        return ColumnRole.NUMERIC, None
    if non_null.astype(str).str.len().mean() >= TEXT_MIN_MEAN_LENGTH:
        return ColumnRole.TEXT, None
    return ColumnRole.CATEGORICAL, None


def _parse_dates(s: pd.Series, force: bool = False) -> pd.Series | None:
    """Parse string dates; return None unless ``force`` or most of a sample parses."""
    text = s.astype(str)
    sample = s.dropna().astype(str).head(200)
    if not force:
        if sample.empty or sample.str.fullmatch(r"\d+(\.\d+)?").all():
            return None
        if not sample.str.contains(r"\d").all():
            return None
    fmt = pd.tseries.api.guess_datetime_format(sample.iloc[0]) if len(sample) else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(text.where(s.notna()), errors="coerce", format=fmt or "mixed")
    if force:
        return parsed
    rate = parsed.loc[sample.index].notna().mean()
    return parsed if rate >= DATETIME_PARSE_THRESHOLD else None


def _numeric_stats(x: pd.Series, with_distribution: bool) -> dict:
    if x.empty:
        return {}
    stats = {"min": float(x.min()), "max": float(x.max())}
    if not with_distribution:
        return stats
    q25, median, q75 = (float(v) for v in x.quantile([0.25, 0.5, 0.75]))
    iqr = q75 - q25
    outliers = ((x < q25 - 1.5 * iqr) | (x > q75 + 1.5 * iqr)).sum() if iqr > 0 else 0
    return stats | {
        "mean": float(x.mean()),
        "std": float(x.std()) if len(x) > 1 else None,
        "q25": q25,
        "median": median,
        "q75": q75,
        "n_zeros": int((x == 0).sum()),
        "n_negative": int((x < 0).sum()),
        "n_outliers": int(outliers),
    }


def profile(df: pd.DataFrame, spec: DatasetSpec) -> DatasetProfile:
    """Profile of a clean time-series frame (see ``analyst.contracts``)."""
    columns = [profile_column(df[col]) for col in df.columns]
    return DatasetProfile(
        name=spec.name,
        n_rows=len(df),
        n_cols=df.shape[1],
        start=df.index.min().to_pydatetime() if len(df) else None,
        end=df.index.max().to_pydatetime() if len(df) else None,
        inferred_freq=infer_freq(df.index),
        columns=columns,
    )


def infer_freq(index: pd.DatetimeIndex) -> str | None:
    if len(index) < 3:
        return None
    freq = pd.infer_freq(index)
    if freq is None:
        freq = pd.tseries.frequencies.to_offset(index.to_series().diff().median()).freqstr
    return freq


def format_profile(p: DatasetProfile) -> str:
    """Human-readable text summary of a profile."""
    lines = [
        f"Dataset: {p.name}  |  {p.n_rows:,} rows x {p.n_cols} columns  |  "
        f"{p.n_duplicate_rows} duplicate rows  |  {p.memory_mb} MB",
    ]
    if p.start:
        lines.append(f"Date range: {p.start:%Y-%m-%d} to {p.end:%Y-%m-%d}")
    lines.append("")
    lines.append(f"{'column':<16}{'role':<13}{'dtype':<10}{'missing':>8}{'unique':>8}  summary")
    for c in p.columns:
        lines.append(
            f"{c.name[:15]:<16}{c.role.value if c.role else '-':<13}{c.dtype[:9]:<10}"
            f"{c.n_missing:>8}{c.n_unique:>8}  {_column_summary(c)}"
        )
    if p.quality:
        lines += ["", f"Quality score: {p.quality.score:.0f}/100"]
        lines += [f"  [{i.severity.value}] {i.message}" for i in p.quality.issues]
    return "\n".join(lines)


def _column_summary(c: ColumnProfile) -> str:
    if c.role is ColumnRole.NUMERIC:
        return f"mean {c.mean:,.2f}, median {c.median:,.2f}, range [{c.min:,.2f}, {c.max:,.2f}]"
    if c.role is ColumnRole.DATETIME and c.min_date:
        return f"{c.min_date:%Y-%m-%d} to {c.max_date:%Y-%m-%d}"
    if c.top_values:
        return ", ".join(f"{v.value[:20]} ({v.count})" for v in c.top_values[:3])
    return ""


def main(argv: list[str] | None = None) -> None:
    from analyst.data.ingest import load_dataset

    parser = argparse.ArgumentParser(description="Profile a CSV/Excel/Parquet dataset.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index")
    parser.add_argument("--json", type=Path, help="write the profile as JSON to this path")
    args = parser.parse_args(argv)

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    result = profile_dataset(load_dataset(args.path, sheet_name=sheet), name=args.path.stem)
    print(format_profile(result))
    if args.json:
        args.json.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nProfile written to {args.json}")


if __name__ == "__main__":
    main()
