"""Individual data-quality checks. Each returns a CheckResult with a row-level failure mask."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

AMOUNT_HINT = r"(amt|amount|value|price|fare|fee|total|spend)"
PREVIOUS_HINT = r"(prev|previous|last|prior)"
CURRENCY_HINT = r"(^|_)(currency|ccy)$"


def name_matches(name: str, pattern: str) -> bool:
    return re.search(pattern, name.lower()) is not None


@dataclass
class CheckResult:
    dimension: str
    name: str
    columns: list
    total: int
    failed: int
    source: str = "auto"                  # auto | config | reference
    unit: str = "rows"
    details: dict = field(default_factory=dict)
    example_rows: list = field(default_factory=list)
    systematic: bool = False
    severity: str = "info"
    mask: pd.Series | None = field(default=None, repr=False)

    @property
    def pass_rate(self) -> float:
        return 1.0 if self.total == 0 else 1 - self.failed / self.total

    def to_dict(self) -> dict:
        return {"dimension": self.dimension, "name": self.name, "columns": self.columns,
                "total": self.total, "failed": self.failed, "pass_rate": round(self.pass_rate, 6),
                "unit": self.unit, "source": self.source, "severity": self.severity,
                "systematic": self.systematic, "details": self.details, "example_rows": self.example_rows}


def _result(dimension, name, columns, mask: pd.Series, **kw) -> CheckResult:
    mask = mask.fillna(False).astype(bool)
    failed = int(mask.sum())
    return CheckResult(dimension=dimension, name=name, columns=columns, total=len(mask), failed=failed,
                       example_rows=[int(i) for i in mask[mask].index[:5]], mask=mask, **kw)


# ---------------- completeness ----------------
def cell_completeness(df: pd.DataFrame) -> CheckResult:
    missing = df.isna()
    per_col = {c: int(v) for c, v in missing.sum().items() if v}
    res = _result("completeness", "missing_values", list(df.columns), missing.any(axis=1),
                  details={"missing_cells": int(missing.values.sum()), "missing_by_column": per_col})
    res.total, res.failed, res.unit = int(df.size), int(missing.values.sum()), "cells"
    return res


def mandatory_fields(df: pd.DataFrame, columns: list, source="auto") -> CheckResult:
    return _result("completeness", "mandatory_fields", columns, df[columns].isna().any(axis=1), source=source,
                   details={"missing_by_column": {c: int(df[c].isna().sum()) for c in columns}})


# ---------------- validity ----------------
def numeric_parseable(s: pd.Series) -> CheckResult:
    mask = pd.to_numeric(s, errors="coerce").isna() & s.notna()
    return _result("validity", f"numeric_parse:{s.name}", [s.name], mask)


def datetime_parseable(raw: pd.Series, parsed: pd.Series) -> CheckResult:
    return _result("validity", f"datetime_parse:{raw.name}", [raw.name], parsed.isna() & raw.notna())


def not_in_future(name: str, parsed: pd.Series, now: pd.Timestamp) -> CheckResult:
    return _result("validity", f"future_datetime:{name}", [name], parsed > now,
                   details={"reference_time": now.isoformat()})


def in_range(s: pd.Series, low, high, source="auto") -> CheckResult:
    x = pd.to_numeric(s, errors="coerce")
    mask = x.notna() & ((x < low) | (x > high))
    return _result("validity", f"range:{s.name}", [s.name], mask, source=source,
                   details={"min_allowed": low, "max_allowed": high})


def allowed_values(s: pd.Series, allowed, source="config") -> CheckResult:
    allowed = set(map(str, allowed))
    mask = s.notna() & ~s.astype(str).isin(allowed)
    unexpected = s[mask].astype(str).value_counts().head(5)
    return _result("validity", f"allowed_values:{s.name}", [s.name], mask, source=source,
                   details={"allowed_count": len(allowed), "unexpected_value_count": int(unexpected.size)})


def matches_pattern(s: pd.Series, pattern: str, source="auto") -> CheckResult:
    mask = s.notna() & ~s.astype(str).str.fullmatch(pattern)
    return _result("validity", f"pattern:{s.name}", [s.name], mask, source=source, details={"pattern": pattern})


def id_format(s: pd.Series, coverage: float = 0.99) -> CheckResult | None:
    """Flag IDs whose shape (letters->a, digits->9, length) differs from the dominant shapes."""
    text = s.dropna().astype(str)
    if text.empty:
        return None
    # Shape = length + character classes. Hex strings form one class, so a random hex ID
    # that happens to contain no letters (e.g. 32 digits) is not mistaken for a malformed ID.
    is_hex = text.str.fullmatch(r"[0-9a-fA-F]+")
    classes = pd.Series(np.where(text.str.contains(r"[A-Za-z]"), "a", ""), index=text.index) \
        + np.where(text.str.contains(r"\d"), "9", "") + np.where(text.str.contains(r"[^A-Za-z0-9]"), "#", "")
    signature = text.str.len().astype(str) + "|" + classes.where(~is_hex, "hex")
    signature = pd.Series(signature, index=text.index)
    counts = signature.value_counts(normalize=True)
    top = counts.head(3)
    if top.sum() < coverage:
        return None            # IDs legitimately vary in shape (e.g. card numbers of different lengths)
    good = set(top[top >= 0.001].index)
    mask = pd.Series(False, index=s.index)
    mask.loc[text.index] = ~signature.isin(good)
    return _result("validity", f"id_format:{s.name}", [s.name], mask,
                   details={"dominant_formats": len(good)})


def binary_target(s: pd.Series) -> CheckResult:
    x = pd.to_numeric(s, errors="coerce")
    return _result("validity", f"target_values:{s.name}", [s.name], s.notna() & ~x.isin([0, 1]))


def plausible_age(name: str, birth: pd.Series, event: pd.Series, low=0, high=120) -> CheckResult:
    age = (event - birth).dt.days / 365.25
    return _result("validity", f"plausible_age:{name}", [name], age.notna() & ((age < low) | (age > high)),
                   details={"min_age": low, "max_age": high})


# ---------------- consistency ----------------
def epoch_matches_datetime(epoch_name: str, epoch: pd.Series, dt_name: str, dt: pd.Series,
                           tolerance_seconds: float = 1.0) -> CheckResult:
    dt_seconds = (dt - pd.Timestamp("1970-01-01")).dt.total_seconds()
    diff = dt_seconds - pd.to_numeric(epoch, errors="coerce")
    valid = diff.dropna()
    details = {}
    if len(valid):
        median = float(valid.median())
        details = {"median_offset_seconds": median, "median_offset_days": round(median / 86400, 2),
                   "offset_is_constant": bool(valid.std() < tolerance_seconds) if len(valid) > 1 else True}
    mask = diff.notna() & (diff.abs() > tolerance_seconds)
    return _result("consistency", f"epoch_vs_datetime:{epoch_name}", [epoch_name, dt_name], mask, details=details)


def ordered(earlier_name: str, earlier: pd.Series, later_name: str, later: pd.Series, source="auto") -> CheckResult:
    return _result("consistency", f"order:{earlier_name}<={later_name}", [earlier_name, later_name],
                   earlier.notna() & later.notna() & (earlier > later), source=source)


def entity_attribute_stability(df: pd.DataFrame, entity: str, candidates: list,
                               min_stable_share: float = 0.95) -> CheckResult | None:
    """Attributes that are fixed for almost every entity (name, birth date, ...) should be fixed for all."""
    if not candidates:
        return None
    nunique = df.groupby(entity, sort=False)[candidates].nunique(dropna=True)
    stable_share = (nunique <= 1).mean()
    stable_cols = [c for c in candidates if stable_share[c] >= min_stable_share]
    if not stable_cols:
        return None
    inconsistent = {c: nunique.index[nunique[c] > 1] for c in stable_cols}
    mask = pd.Series(False, index=df.index)
    for c, ents in inconsistent.items():
        if len(ents):
            # Flag only rows that disagree with their entity's most common value.
            sub = df.loc[df[entity].isin(ents), [entity, c]]
            mode = sub.groupby(entity)[c].agg(lambda v: v.mode().iloc[0] if v.notna().any() else np.nan)
            expected = sub[entity].map(mode)
            mask.loc[sub.index] |= sub[c].notna() & sub[c].ne(expected)
    return _result("consistency", f"entity_attribute_stability:{entity}", [entity] + stable_cols, mask,
                   details={"stable_attributes": stable_cols,
                            "inconsistent_entities_by_attribute": {c: int(len(e)) for c, e in inconsistent.items()}})


# ---------------- uniqueness ----------------
def duplicate_ids(s: pd.Series) -> CheckResult:
    mask = s.notna() & s.duplicated(keep=False)
    return _result("uniqueness", f"duplicate_id:{s.name}", [s.name], mask,
                   details={"distinct_ids_repeated": int(s[mask].nunique())})


def duplicate_rows(df: pd.DataFrame, ignore: list, name: str) -> CheckResult:
    cols = [c for c in df.columns if c not in ignore]
    return _result("uniqueness", name, cols, df.duplicated(subset=cols, keep="first"),
                   details={"ignored_columns": ignore})


def repeated_transactions(df: pd.DataFrame, entity: str, time: pd.Series, keys: list,
                          window_seconds: float = 60) -> CheckResult:
    """Same entity and same key values (e.g. amount, merchant) again within a short time window."""
    frame = df[[entity] + keys].copy()
    frame["_t"] = time
    frame = frame.dropna(subset=["_t"]).sort_values([entity, "_t"])
    same = pd.Series(True, index=frame.index)
    for c in [entity] + keys:
        same &= frame[c].eq(frame[c].shift())
    gap = frame["_t"].diff().dt.total_seconds()
    repeat = same & gap.le(window_seconds)
    mask = pd.Series(False, index=df.index)
    mask.loc[repeat[repeat].index] = True
    return _result("uniqueness", f"repeated_transactions:{entity}", [entity] + keys, mask,
                   details={"window_seconds": window_seconds, "keys": keys})


# ---------------- integrity ----------------
def referential(s: pd.Series, reference_ids, reference_name: str) -> CheckResult:
    ref = pd.Index(pd.Series(list(reference_ids)).astype(str).unique())
    mask = s.notna() & ~s.astype(str).isin(ref)
    return _result("integrity", f"foreign_key:{s.name}->{reference_name}", [s.name], mask, source="reference",
                   details={"reference": reference_name, "orphan_distinct_values": int(s[mask].nunique())})
