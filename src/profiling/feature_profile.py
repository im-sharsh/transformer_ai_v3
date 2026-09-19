"""Dataset structural profiling and evidence-based feature-family recommendation.

This module deliberately does *not* inspect downstream model scores. It measures whether
transactional history features are structurally supported by the source data, then emits
an auditable recommendation that callers may accept or override.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureProfile:
    rows: int
    entity_column: str | None
    time_column: str | None
    amount_column: str | None
    counterparty_column: str | None
    category_column: str | None
    entity_unique_ratio: float | None
    entity_repeat_row_share: float | None
    entity_rows_median: float | None
    entity_rows_p90: float | None
    entity_rows_max: int | None
    entity_sequence_coverage: float | None
    counterparty_unique_ratio: float | None
    counterparty_repeat_row_share: float | None
    temporal_span_hours: float | None
    simultaneous_row_share: float | None
    category_cardinality: int | None
    amount_missing_share: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureRecommendation:
    level: str
    numeric_mode: str
    feature_groups: tuple[str, ...]
    confidence: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: FeatureProfile

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["feature_groups"] = list(self.feature_groups)
        out["reasons"] = list(self.reasons)
        out["warnings"] = list(self.warnings)
        return out


def _repeat_stats(s: pd.Series) -> tuple[float, float, float, float, int] | tuple[None, None, None, None, None]:
    x = s.dropna()
    if x.empty:
        return None, None, None, None, None
    counts = x.value_counts(dropna=False)
    unique_ratio = float(counts.size / len(x))
    repeated_values = set(counts[counts >= 2].index)
    repeat_row_share = float(x.isin(repeated_values).mean())
    return unique_ratio, repeat_row_share, float(counts.median()), float(counts.quantile(0.90)), int(counts.max())


def _sequence_coverage(df: pd.DataFrame, entity: str, time: str, sequence_length: int) -> float | None:
    if entity not in df or time not in df or sequence_length <= 0:
        return None
    work = df[[entity, time]].copy()
    work[time] = pd.to_datetime(work[time], errors="coerce")
    work = work.dropna()
    if work.empty:
        return None
    # Strictly earlier timestamps only. Rows at the same entity+timestamp are peers and
    # do not count as prior observations for each other.
    group_sizes = work.groupby([entity, time], sort=False).size().rename("same_time_rows").reset_index()
    group_sizes = group_sizes.sort_values([entity, time], kind="stable")
    group_sizes["prior_rows"] = group_sizes.groupby(entity)["same_time_rows"].cumsum() - group_sizes["same_time_rows"]
    eligible = group_sizes["prior_rows"] >= sequence_length
    eligible_rows = int(group_sizes.loc[eligible, "same_time_rows"].sum())
    return float(eligible_rows / len(work))


def profile_feature_support(
    df: pd.DataFrame,
    *,
    entity: str | None,
    time: str | None,
    amount: str | None,
    counterparty: str | None = None,
    category: str | None = None,
    sequence_length: int = 3,
) -> FeatureProfile:
    """Measure structural support for temporal/history/sequence feature families."""
    n = len(df)
    if entity and entity in df:
        eu, er, emed, ep90, emax = _repeat_stats(df[entity])
    else:
        eu = er = emed = ep90 = emax = None
    if counterparty and counterparty in df:
        cu, cr, _, _, _ = _repeat_stats(df[counterparty])
    else:
        cu = cr = None

    seq = _sequence_coverage(df, entity, time, sequence_length) if entity and time else None

    span_hours = simultaneous = None
    if time and time in df:
        t = pd.to_datetime(df[time], errors="coerce")
        valid = t.dropna()
        if not valid.empty:
            span_hours = float((valid.max() - valid.min()).total_seconds() / 3600.0)
            # Rows sharing their timestamp with >=1 other row. This is informational:
            # point-in-time code already treats them as simultaneous peers.
            counts = valid.value_counts()
            repeated_times = set(counts[counts >= 2].index)
            simultaneous = float(valid.isin(repeated_times).mean())

    cat_card = int(df[category].nunique(dropna=True)) if category and category in df else None
    amount_missing = float(pd.to_numeric(df[amount], errors="coerce").isna().mean()) if amount and amount in df else None

    return FeatureProfile(
        rows=n,
        entity_column=entity if entity in df.columns else None,
        time_column=time if time in df.columns else None,
        amount_column=amount if amount in df.columns else None,
        counterparty_column=counterparty if counterparty in df.columns else None,
        category_column=category if category in df.columns else None,
        entity_unique_ratio=eu,
        entity_repeat_row_share=er,
        entity_rows_median=emed,
        entity_rows_p90=ep90,
        entity_rows_max=emax,
        entity_sequence_coverage=seq,
        counterparty_unique_ratio=cu,
        counterparty_repeat_row_share=cr,
        temporal_span_hours=span_hours,
        simultaneous_row_share=simultaneous,
        category_cardinality=cat_card,
        amount_missing_share=amount_missing,
    )


def recommend_feature_profile(
    profile: FeatureProfile,
    *,
    min_history_repeat_share: float = 0.25,
    min_sequence_coverage: float = 0.10,
    min_counterparty_repeat_share: float = 0.25,
) -> FeatureRecommendation:
    """Recommend E1/E2 feature families from structural coverage only.

    Conservative policy:
    * E1 continuous is the fallback when behavioral evidence is weak.
    * Temporal features are considered only when a usable event-time column exists.
    * History requires repeated entities; counterparty repetition strengthens confidence
      but cannot independently enable history because today's E2 history builder is
      entity-anchored.
    * Sequence requires a material share of rows with at least ``sequence_length``
      strictly earlier observations for the same entity.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    groups: list[str] = []

    has_time = profile.time_column is not None and profile.temporal_span_hours is not None
    has_amount = profile.amount_column is not None and (profile.amount_missing_share or 0.0) < 0.50
    has_entity = profile.entity_column is not None

    if has_time:
        groups.append("temporal")
        reasons.append("usable event time is available, so leakage-safe temporal context can be generated")
    else:
        warnings.append("no usable event-time column; temporal/history/sequence features are unsafe or unavailable")

    repeat_share = profile.entity_repeat_row_share or 0.0
    seq_cov = profile.entity_sequence_coverage or 0.0
    cp_repeat = profile.counterparty_repeat_row_share or 0.0

    if has_entity and has_time and has_amount and repeat_share >= min_history_repeat_share:
        groups.append("history")
        reasons.append(f"{repeat_share:.1%} of entity rows belong to repeated entities (threshold {min_history_repeat_share:.0%})")
        if cp_repeat >= min_counterparty_repeat_share:
            reasons.append(f"counterparty repetition is also substantial ({cp_repeat:.1%})")
    elif has_entity:
        warnings.append(
            f"entity history support is weak ({repeat_share:.1%} repeated-row coverage; threshold {min_history_repeat_share:.0%})"
        )

    if has_entity and has_time and has_amount and seq_cov >= min_sequence_coverage:
        groups.append("sequence")
        reasons.append(f"{seq_cov:.1%} of rows have enough strictly earlier entity history for sequence features")
    elif has_entity:
        warnings.append(
            f"sequence support is weak ({seq_cov:.1%} eligible-row coverage; threshold {min_sequence_coverage:.0%})"
        )

    # Temporal-only E2 did not consistently beat E1 in our validated experiments.
    # Therefore require at least one behavioral family before recommending E2.
    behavioral = any(g in groups for g in ("history", "sequence"))
    if behavioral:
        confidence = "high" if "history" in groups and "sequence" in groups else "medium"
        return FeatureRecommendation(
            level="E2",
            numeric_mode="quantile_bin",
            feature_groups=tuple(groups),
            confidence=confidence,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            metrics=profile,
        )

    reasons.append("behavioral coverage is insufficient to justify E2 automatically; use conservative continuous E1")
    return FeatureRecommendation(
        level="E1",
        numeric_mode="continuous",
        feature_groups=(),
        confidence="high" if has_time else "medium",
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        metrics=profile,
    )


def recommend_for_autodata(df: pd.DataFrame, roles: Any, config: dict[str, Any]) -> FeatureRecommendation:
    """Bridge AutoData's inferred ``Roles`` object to the adaptive recommendation policy."""
    acfg = config.get("adaptive_features", {})
    seq_len = int(config.get("features", {}).get("sequence_length", 3))
    p = profile_feature_support(
        df,
        entity=getattr(roles, "entity", None),
        time=getattr(roles, "time", None),
        amount=getattr(roles, "amount", None),
        counterparty=getattr(roles, "merchant", None),
        category=getattr(roles, "category", None),
        sequence_length=seq_len,
    )
    return recommend_feature_profile(
        p,
        min_history_repeat_share=float(acfg.get("min_history_repeat_share", 0.25)),
        min_sequence_coverage=float(acfg.get("min_sequence_coverage", 0.10)),
        min_counterparty_repeat_share=float(acfg.get("min_counterparty_repeat_share", 0.25)),
    )


def apply_feature_recommendation(config: dict[str, Any], recommendation: FeatureRecommendation) -> dict[str, Any]:
    """Return a deep-copied config with the recommendation applied; original config is unchanged."""
    import copy
    out = copy.deepcopy(config)
    out.setdefault("representation", {})["numeric_mode"] = recommendation.numeric_mode
    if recommendation.level == "E2":
        out.setdefault("features", {})["enabled_groups"] = list(recommendation.feature_groups)
    return out
