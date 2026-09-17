"""Duplicate detection with explicit policies. Nothing is deleted silently."""
from __future__ import annotations

import pandas as pd

POLICIES = ("remove", "quarantine", "retain")


def find_duplicates(df: pd.DataFrame, subset: list) -> pd.Series:
    """True for every copy after the first occurrence."""
    return df.duplicated(subset=subset, keep="first")


def conflicting_ids(df: pd.DataFrame, id_column: str) -> pd.Series:
    """True for all rows sharing an ID (after exact duplicates have been handled)."""
    return df[id_column].notna() & df[id_column].duplicated(keep=False)
