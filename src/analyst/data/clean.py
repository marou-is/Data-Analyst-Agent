from __future__ import annotations

import pandas as pd

from analyst.contracts import TIMESTAMP, CleaningStep, CleanResult, ColumnRole, DatasetProfile, DatasetSpec
from analyst.data.profile import _parse_dates

_NUMBER_JUNK = r"[$€£%,\s]"


def clean(raw: pd.DataFrame, spec: DatasetSpec) -> CleanResult:
    """Turn a raw frame into a clean frame (see ``analyst.contracts``)."""
    steps: list[CleaningStep] = []
    df = raw[[spec.timestamp_col, *spec.target_cols]].copy()
    rows_before = len(df)

    df[spec.timestamp_col] = pd.to_datetime(df[spec.timestamp_col], errors="coerce")
    bad_ts = int(df[spec.timestamp_col].isna().sum())
    if bad_ts:
        df = df.dropna(subset=[spec.timestamp_col])
        steps.append(CleaningStep(name="drop_bad_timestamps", description="Dropped rows with unparseable timestamps", rows_affected=bad_ts))

    for col in spec.target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df = df.sort_values(spec.timestamp_col)
    dupes = int(df[spec.timestamp_col].duplicated().sum())
    if dupes:
        df = df.drop_duplicates(subset=[spec.timestamp_col], keep="first")
        steps.append(CleaningStep(name="drop_duplicate_timestamps", description="Kept first row per duplicated timestamp", rows_affected=dupes))

    df = df.set_index(spec.timestamp_col).rename_axis(TIMESTAMP)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    n_missing = int(df[spec.target_cols].isna().sum().sum())
    if n_missing:
        df[spec.target_cols] = df[spec.target_cols].interpolate(method="time", limit_direction="both")
        steps.append(CleaningStep(name="interpolate_missing", description="Time-interpolated missing target values", rows_affected=n_missing))

    return CleanResult(data=df, steps=steps, rows_before=rows_before, rows_after=len(df))


def clean_table(raw: pd.DataFrame, profile: DatasetProfile) -> CleanResult:
    """Turn a raw frame into a *clean table* (see ``analyst.contracts``).

    Uses the column roles and quality issues of ``profile`` (from ``profile_dataset``):
    parses datetime-role columns, coerces numeric-role columns, zero-pads code columns
    flagged ``lost_leading_zeros``, strips whitespace in categorical/text columns,
    unifies categories differing only by case/whitespace, and drops full and
    near-duplicate rows (keeping the first). Rows with missing values are kept.
    """
    df = raw.copy()
    steps: list[CleaningStep] = []
    roles = {c.name: c.role for c in profile.columns if c.name in df.columns}

    for col in [c for c, r in roles.items() if r is ColumnRole.DATETIME]:
        steps += _parse_datetime(df, col)
    for col in [c for c, r in roles.items() if r is ColumnRole.NUMERIC]:
        steps += _coerce_numeric(df, col)
    padded = {i.column for i in profile.quality.issues if i.check == "lost_leading_zeros"} if profile.quality else set()
    for col in [c for c in roles if c in padded]:
        steps += _pad_leading_zeros(df, col)
    for col, role in roles.items():
        if role in (ColumnRole.CATEGORICAL, ColumnRole.TEXT):
            steps += _normalize_text(df, col, unify=role is ColumnRole.CATEGORICAL)

    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df[~df.duplicated(keep="first")]
        steps.append(CleaningStep(name="drop_duplicate_rows", description=f"Dropped {n_dup} fully duplicated rows",
                                  rows_affected=n_dup))
    keys = [c.name for c in profile.columns if c.name in roles and c.role is ColumnRole.IDENTIFIER
            and c.pct_unique == 1 and c.n_missing == 0]
    rest = [c for c in df.columns if c not in keys]
    if keys and rest:
        near = df.duplicated(subset=rest, keep="first")
        if near.any():
            df = df[~near]
            steps.append(CleaningStep(
                name="drop_near_duplicate_rows", rows_affected=int(near.sum()),
                description=f"Dropped {int(near.sum())} rows identical to an earlier row except for the unique key {keys}",
            ))
    return CleanResult(data=df, steps=steps, rows_before=len(raw), rows_after=len(df))


def _parse_datetime(df: pd.DataFrame, col: str) -> list[CleaningStep]:
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return []
    parsed = _parse_dates(s, force=True)
    df[col] = parsed
    n_bad = int((s.notna() & parsed.isna()).sum())
    note = f"; {n_bad} unparseable values set to missing" if n_bad else ""
    return [CleaningStep(name="parse_datetime", description=f"Parsed {col!r} as datetime{note}",
                         rows_affected=int(s.notna().sum()))]


def _coerce_numeric(df: pd.DataFrame, col: str) -> list[CleaningStep]:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return []
    text = s.astype(str).str.replace(_NUMBER_JUNK, "", regex=True).where(s.notna())
    numbers = pd.to_numeric(text, errors="coerce")
    df[col] = numbers
    n_bad = int((s.notna() & numbers.isna()).sum())
    note = f"; {n_bad} non-numeric values set to missing" if n_bad else ""
    return [CleaningStep(name="coerce_numeric", description=f"Converted {col!r} to numbers{note}",
                         rows_affected=int(s.notna().sum()))]


def _pad_leading_zeros(df: pd.DataFrame, col: str) -> list[CleaningStep]:
    s = df[col]
    if not pd.api.types.is_integer_dtype(s):
        return []
    digits = s.dropna().astype("int64").astype(str)
    width = int(digits.str.len().max())
    n_short = int((digits.str.len() < width).sum())
    df[col] = digits.str.zfill(width).reindex(s.index)
    return [CleaningStep(name="pad_leading_zeros", rows_affected=n_short,
                         description=f"Stored code column {col!r} as text, zero-padded to {width} digits")]


def _normalize_text(df: pd.DataFrame, col: str, unify: bool) -> list[CleaningStep]:
    s = df[col]
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    is_text = s.map(lambda v: isinstance(v, str)).astype(bool)
    if not is_text.any():
        return []
    original = s[is_text].astype(str)
    stripped = original.str.strip()
    steps = []
    n_stripped = int((stripped != original).sum())
    if n_stripped:
        steps.append(CleaningStep(name="strip_whitespace", rows_affected=n_stripped,
                                  description=f"Stripped surrounding whitespace in {col!r}"))
    values = stripped
    if unify:
        values = _unify_spelling(stripped)
        n_unified = int((values != stripped).sum())
        if n_unified:
            steps.append(CleaningStep(
                name="unify_categories", rows_affected=n_unified,
                description=f"Unified categories of {col!r} differing only by case to the most frequent spelling",
            ))
    if steps:
        out = s.copy()
        out[is_text] = values
        df[col] = out
    return steps


def _unify_spelling(values: pd.Series) -> pd.Series:
    keys = values.str.lower()
    if keys.nunique() == values.nunique():
        return values
    counts = values.value_counts().rename_axis("spelling").reset_index()
    counts["key"] = counts["spelling"].str.lower()
    best = counts.sort_values(["count", "spelling"], ascending=[False, True]).drop_duplicates("key")
    return keys.map(best.set_index("key")["spelling"])
