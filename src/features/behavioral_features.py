"""Point-in-time history features for transaction data.

Every feature for a transaction uses only transactions with an event timestamp *strictly earlier* than the
current transaction. Rows sharing the exact same timestamp are treated as simultaneous and cannot see one
another. Past fraud labels are never used: in practice labels can arrive much later.

Column names come from HistoryConfig, so nothing here is specific to one dataset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class HistoryConfig:
    entity: str                               # e.g. card / account / customer id
    time: str                                 # event timestamp
    amount: str
    category: str | None = None
    merchant: str | None = None
    windows: tuple = ("1h", "24h", "7d", "30d")
    sequence_length: int = 5

    def to_dict(self):
        return asdict(self)


# Features safe to compute but not exposed to the model by default. The cumulative count grows monotonically
# with calendar time and can become a temporal shortcut under distribution shift.
DIAGNOSTIC_ONLY = ["card_txn_count_before"]


def _sorted(frame: pd.DataFrame, keys: list) -> pd.DataFrame:
    return frame.sort_values(keys, kind="mergesort")


def _window_counts(f: pd.DataFrame, key: str, time: str, amount: pd.Series, windows) -> dict:
    """Counts and amount sums in [t - w, t) per key; f must be sorted by [key, time].

    closed='left' is deliberate: all rows at the current timestamp are excluded, including duplicate timestamps.
    """
    tmp = pd.DataFrame({"k": f[key].to_numpy(), "one": 1.0, "amt": amount.to_numpy()},
                       index=pd.DatetimeIndex(f[time].to_numpy()))
    out = {}
    grouped = tmp.groupby("k", sort=False)
    for w in windows:
        r = grouped.rolling(w, closed="left")
        out[f"count_{w}"] = np.nan_to_num(r["one"].sum().to_numpy(), nan=0.0)
        out[f"amount_sum_{w}"] = np.nan_to_num(r["amt"].sum().to_numpy(), nan=0.0)
    return out


def _strict_prior_count(f: pd.DataFrame, key_cols: list[str], time_values: pd.Series) -> pd.Series:
    """Number of rows in key_cols with timestamp strictly earlier than the current timestamp."""
    total_before_row = f.groupby(key_cols, sort=False).cumcount().astype("float64")
    within_timestamp_before_row = f.groupby(key_cols + [time_values], sort=False).cumcount().astype("float64")
    return total_before_row - within_timestamp_before_row


def _strict_prior_sum(values: pd.Series, key_values: list[pd.Series], time_values: pd.Series) -> pd.Series:
    """Sum of values in the same key with timestamp strictly earlier than the current timestamp."""
    total = values.groupby(key_values, sort=False).cumsum()
    within_timestamp = values.groupby(key_values + [time_values], sort=False).cumsum()
    return total - within_timestamp


def _previous_distinct_time(f: pd.DataFrame, key: str, time_values: pd.Series) -> pd.Series:
    """Most recent timestamp strictly before the current timestamp for each key."""
    blocks = pd.DataFrame({"key": f[key].to_numpy(), "time": time_values.to_numpy()}).drop_duplicates()
    blocks["previous_time"] = blocks.groupby("key", sort=False)["time"].shift(1)
    mapping = pd.Series(
        blocks["previous_time"].to_numpy(),
        index=pd.MultiIndex.from_arrays([blocks["key"], blocks["time"]]),
    )
    keys = pd.MultiIndex.from_arrays([f[key].to_numpy(), time_values.to_numpy()])
    return pd.Series(mapping.reindex(keys).to_numpy(), index=f.index)


def card_history(frame: pd.DataFrame, cfg: HistoryConfig) -> pd.DataFrame:
    e, t, a = cfg.entity, cfg.time, cfg.amount
    f = _sorted(frame, [e, t])
    f_time = pd.to_datetime(f[t])
    amt = f[a].astype("float64")

    n = _strict_prior_count(f, [e], f_time)
    s1 = _strict_prior_sum(amt, [f[e]], f_time)
    s2 = _strict_prior_sum(amt ** 2, [f[e]], f_time)
    mean_prev = (s1 / n).where(n > 0)
    var_prev = ((s2 - n * mean_prev ** 2) / (n - 1)).where(n > 1).clip(lower=0)
    std_prev = np.sqrt(var_prev)

    out = pd.DataFrame(index=f.index)
    out["card_txn_count_before"] = n
    prev_distinct = pd.to_datetime(_previous_distinct_time(f, e, f_time))
    out["card_seconds_since_prev"] = (f_time - prev_distinct).dt.total_seconds()

    for name, values in _window_counts(f.assign(**{t: f_time}), e, t, amt, cfg.windows).items():
        out[f"card_{name}"] = values

    out["card_amount_mean_before"] = mean_prev
    out["card_amount_std_before"] = std_prev
    out["card_amount_ratio_to_mean"] = (amt / mean_prev).where(mean_prev > 0)
    out["card_amount_zscore"] = ((amt - mean_prev) / std_prev).where((n > 1) & (std_prev > 0))

    if cfg.category:
        prev_same_cat = _strict_prior_count(f, [e, cfg.category], f_time)
        out["card_first_time_category"] = (prev_same_cat == 0).astype("float64")
        out["card_category_share_before"] = (prev_same_cat / n).where(n > 0)
    if cfg.merchant:
        prev_same_merchant = _strict_prior_count(f, [e, cfg.merchant], f_time)
        out["card_first_time_merchant"] = (prev_same_merchant == 0).astype("float64")
    return out.reindex(frame.index)


def merchant_history(frame: pd.DataFrame, cfg: HistoryConfig) -> pd.DataFrame:
    m, t, a = cfg.merchant, cfg.time, cfg.amount
    f = _sorted(frame, [m, t])
    f_time = pd.to_datetime(f[t])
    amt = f[a].astype("float64")
    n = _strict_prior_count(f, [m], f_time)
    prior_sum = _strict_prior_sum(amt, [f[m]], f_time)
    mean_prev = (prior_sum / n).where(n > 0)

    out = pd.DataFrame(index=f.index)
    counts = _window_counts(f.assign(**{t: f_time}), m, t, amt, ("24h", "7d"))
    out["merchant_count_24h"] = counts["count_24h"]
    out["merchant_count_7d"] = counts["count_7d"]
    out["merchant_amount_mean_before"] = mean_prev
    out["merchant_amount_ratio_to_mean"] = (amt / mean_prev).where(mean_prev > 0)
    return out.reindex(frame.index)


def _lookup_prior_rows(f: pd.DataFrame, entity: str, time_values: pd.Series, value: pd.Series, k: int) -> pd.Series:
    """Value from the k-th most recent transaction whose timestamp is strictly earlier than the current row."""
    position = f.groupby(entity, sort=False).cumcount().astype("int64")
    within_time = f.groupby([entity, time_values], sort=False).cumcount().astype("int64")
    block_start = position - within_time

    lookup = pd.Series(
        value.to_numpy(),
        index=pd.MultiIndex.from_arrays([f[entity].to_numpy(), position.to_numpy()]),
    )
    wanted = pd.MultiIndex.from_arrays([f[entity].to_numpy(), (block_start - int(k)).to_numpy()])
    return pd.Series(lookup.reindex(wanted).to_numpy(), index=f.index)


def previous_transactions(frame: pd.DataFrame, cfg: HistoryConfig, hour_col: str | None = None) -> pd.DataFrame:
    """The entity's previous K transactions, excluding every transaction at the current timestamp.

    For simultaneous transactions, each row sees the same history ending immediately before that timestamp.
    """
    e, t, a = cfg.entity, cfg.time, cfg.amount
    f = _sorted(frame, [e, t])
    f_time = pd.to_datetime(f[t])
    hour = f[hour_col] if hour_col else f_time.dt.hour
    out = pd.DataFrame(index=f.index)

    for k in range(1, cfg.sequence_length + 1):
        prev_time = pd.to_datetime(_lookup_prior_rows(f, e, f_time, f_time, k))
        out[f"prev{k}_hours_ago"] = (f_time - prev_time).dt.total_seconds() / 3600
        out[f"prev{k}_amount"] = _lookup_prior_rows(f, e, f_time, f[a], k)
        out[f"prev{k}_hour"] = _lookup_prior_rows(f, e, f_time, hour, k)
        if cfg.category:
            out[f"prev{k}_category"] = _lookup_prior_rows(f, e, f_time, f[cfg.category], k)
    return out.reindex(frame.index)


def format_previous_transactions(prev: pd.DataFrame, k: int) -> pd.Series:
    """Compact text for a language model, e.g. '1) 3.2h ago $12.50 grocery_pos at 14h; 2) ...' or 'none'."""
    parts = []
    for i in range(1, k + 1):
        h, amt, hr = prev[f"prev{i}_hours_ago"], prev[f"prev{i}_amount"], prev[f"prev{i}_hour"]
        cat = prev.get(f"prev{i}_category")
        text = (f"{i}) " + h.map(lambda v: f"{v:.1f}h ago" if pd.notna(v) else "")
                + " $" + amt.map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
                + ((" " + cat.astype(str)) if cat is not None else "")
                + " at " + hr.map(lambda v: f"{int(v)}h" if pd.notna(v) else ""))
        parts.append(text.where(h.notna(), ""))
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.str.cat(p, sep="; ").str.replace(r"(; )+$", "", regex=True)
    return joined.str.strip("; ").replace("", "none")


def build_history_features(frame: pd.DataFrame, cfg: HistoryConfig, hour_col: str | None = None) -> pd.DataFrame:
    parts = [card_history(frame, cfg)]
    if cfg.merchant:
        parts.append(merchant_history(frame, cfg))
    parts.append(previous_transactions(frame, cfg, hour_col))
    return pd.concat(parts, axis=1)


def check_point_in_time(df: pd.DataFrame, feature_fn, time_col: str, group_col: str, n_samples: int = 200,
                        seed: int = 0, atol: float = 1e-6) -> dict:
    """Recompute sampled rows from strictly earlier group history plus the current row only.

    This is intentionally stricter than row-order truncation: rows sharing the current timestamp are excluded.
    Therefore a feature that accidentally treats a simultaneous transaction as historical will fail this check.
    """
    order = df.sort_values([group_col, time_col], kind="mergesort")
    full = feature_fn(order)
    picks = order.sample(min(n_samples, len(order)), random_state=seed).index
    mismatches = {c: 0 for c in full.columns}
    examples = []

    for idx in picks:
        g = order.at[idx, group_col]
        current_time = pd.to_datetime(order.at[idx, time_col])
        group = order[order[group_col] == g]
        times = pd.to_datetime(group[time_col])
        history = group[(times < current_time) | (group.index == idx)]
        recomputed = feature_fn(history).loc[idx]
        for c in full.columns:
            a, b = full.at[idx, c], recomputed[c]
            if pd.isna(a) or pd.isna(b):
                same = pd.isna(a) and pd.isna(b)
            elif isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                same = bool(np.isclose(float(a), float(b), atol=atol))
            else:
                same = a == b
            if not same:
                mismatches[c] += 1
                if len(examples) < 5:
                    examples.append({"row": str(idx), "feature": c, "full_data_value": a,
                                     "strict_history_value": b})
    leaking = [c for c, n in mismatches.items() if n]
    return {"rows_checked": int(len(picks)), "mismatches": mismatches, "leaking_features": leaking,
            "passed": not leaking, "examples": examples, "same_timestamp_policy": "strictly_earlier_only"}
