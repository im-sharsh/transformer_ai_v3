"""Stateless time fields (need no statistics, so no leakage). Named to avoid shadowing Python's datetime."""
from __future__ import annotations

import pandas as pd

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def add_time_features(df: pd.DataFrame, event_col: str, birth_cols: list | None = None) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    t = pd.to_datetime(df[event_col], errors="coerce")
    new = {"hour": t.dt.hour, "day_of_week": t.dt.dayofweek, "month": t.dt.month,
           "is_weekend": (t.dt.dayofweek >= 5).astype("float").where(t.notna())}
    for b in birth_cols or []:
        born = pd.to_datetime(df[b], errors="coerce")
        years = t.dt.year - born.dt.year - ((t.dt.month < born.dt.month) |
                                            ((t.dt.month == born.dt.month) & (t.dt.day < born.dt.day)))
        new["age_years" if len(birth_cols) == 1 else f"age_years_from_{b}"] = years
    for k, v in new.items():
        df[k] = v.astype("Float64")
    return df, list(new)
