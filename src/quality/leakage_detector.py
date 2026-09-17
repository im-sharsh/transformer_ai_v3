"""Leakage detector: flags features that may carry information unavailable at prediction time.

Nothing is removed. Every finding reports feature, leakage_risk, check, evidence, reason and
recommendation, so a human decides and the decision is logged.

Checks
  1. target_name            name suggests the target or its outcome (chargeback, dispute, label, ...)
  2. target_copy            feature reproduces the target almost perfectly
  3. suspiciously_predictive single feature predicts the target on a *later* period extremely well
  4. temporal_instability   predictive under random folds but not on a later period
                            (memorises dates, entities or bursts instead of generalisable patterns)
  5. post_event_time        a timestamp lies after the event being predicted
  6. identity_proxy         entity id or a column fixed per entity with many values
  7. target_word_in_text    text values contain target words (differently per class = leak; everywhere = noise)
  8. cold_start_artifact    target rate for entities never seen before is extreme
  9. entity_concentration   positives concentrated in few entities (context for 6 and 8)
 + check_point_in_time()    verifies engineered history features use only past rows
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET_WORDS = r"(?:fraud|chargeback|charge_back|dispute|label|outcome|confirmed|investigat|reversal|refund)"
POST_EVENT_WORDS = r"(?:settle|resolution|resolved|review|after|final_status|closed|reported)"
RISK_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass
class Finding:
    feature: str
    check: str
    leakage_risk: str
    evidence: dict
    reason: str
    recommendation: str


@dataclass
class LeakageReport:
    dataset_id: str
    target: str
    created_at: str
    seconds: float
    split: dict
    findings: list
    univariate: list = field(default_factory=list)

    def by_feature(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(columns=["feature", "leakage_risk", "checks", "reason", "recommendation"])
        df = pd.DataFrame([asdict(f) for f in self.findings])
        df["_order"] = df["leakage_risk"].map(RISK_ORDER)
        return df.sort_values(["_order", "feature"]).drop(columns="_order")

    def to_dict(self) -> dict:
        return {"dataset_id": self.dataset_id, "target": self.target, "created_at": self.created_at,
                "seconds": self.seconds, "split": self.split,
                "risk_counts": pd.Series([f.leakage_risk for f in self.findings]).value_counts().to_dict(),
                "findings": [asdict(f) for f in self.findings], "univariate": self.univariate}

    def save(self, out_dir: str | Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {"json": out_dir / f"leakage_{self.dataset_id}.json", "md": out_dir / f"leakage_{self.dataset_id}.md"}
        paths["json"].write_text(json.dumps(self.to_dict(), indent=2, default=str))
        paths["md"].write_text(self.to_markdown())
        return paths

    def to_markdown(self) -> str:
        lines = [f"# Leakage report: {self.dataset_id}", "",
                 f"Target `{self.target}`; fit period {self.split['fit_rows']:,} rows, later period "
                 f"{self.split['holdout_rows']:,} rows (split at {self.split['cutoff']}).", "",
                 "Nothing was removed. Each finding needs a human decision.", "",
                 "| Feature | Risk | Check | Reason | Recommendation |", "|---|---|---|---|---|"]
        for f in sorted(self.findings, key=lambda f: (RISK_ORDER[f.leakage_risk], f.feature)):
            lines.append(f"| {f.feature} | **{f.leakage_risk}** | {f.check} | {f.reason} | {f.recommendation} |")
        if self.univariate:
            lines += ["", "## Single-feature predictive power", "",
                      "| Feature | Kind | ROC-AUC random folds (fit period) | ROC-AUC later period | Gap |", "|---|---|---|---|---|"]
            for u in self.univariate:
                lines.append(f"| {u['feature']} | {u['kind']} | {u['auc_random_folds']:.3f} | {u['auc_later_period']:.3f} | "
                             f"{u['auc_random_folds'] - u['auc_later_period']:+.3f} |")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ helpers
def _auc(y, score) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    a = roc_auc_score(y, score)
    return float(max(a, 1 - a))


def _keys(s: pd.Series, kind: str, edges=None):
    """Map a feature to discrete keys: quantile bins for numbers/times, the value itself for categories."""
    if kind == "numeric":
        x = pd.to_numeric(s, errors="coerce").astype("float64")
        keys = pd.Series(np.searchsorted(edges, x, side="right"), index=s.index).astype("float64")
        return keys.where(x.notna(), -1.0)
    return s.astype("string").fillna("__missing__")


def _encode(train_keys: pd.Series, train_y: pd.Series, apply_keys: pd.Series, prior: float, m: float = 20.0) -> np.ndarray:
    stats = pd.DataFrame({"k": train_keys.values, "y": train_y.values}).groupby("k")["y"].agg(["sum", "count"])
    rate = (stats["sum"] + m * prior) / (stats["count"] + m)
    return apply_keys.map(rate).fillna(prior).to_numpy(float)


def univariate_power(fit: pd.DataFrame, later: pd.DataFrame, feature: str, kind: str, target: str,
                     n_bins: int = 50, folds: int = 5, seed: int = 0) -> dict:
    """Smoothed target-rate model of one feature: random-fold AUC in the fit period vs AUC on the later period."""
    y_fit, y_later = fit[target].astype(int), later[target].astype(int)
    prior = float(y_fit.mean())
    edges = None
    if kind == "numeric":
        x = pd.to_numeric(fit[feature], errors="coerce").dropna()
        edges = np.unique(x.quantile(np.linspace(0, 1, n_bins + 1)[1:-1]).values) if len(x) else np.array([])
    k_fit, k_later = _keys(fit[feature], kind, edges), _keys(later[feature], kind, edges)
    fold = np.random.default_rng(seed).integers(0, folds, len(fit))
    oof = np.empty(len(fit))
    for f in range(folds):
        tr, te = fold != f, fold == f
        oof[te] = _encode(k_fit[tr], y_fit[tr], k_fit[te], prior)
    later_score = _encode(k_fit, y_fit, k_later, prior)
    return {"feature": feature, "kind": kind, "auc_random_folds": _auc(y_fit, oof),
            "auc_later_period": _auc(y_later, later_score),
            "later_values_unseen_share": float((~k_later.isin(set(k_fit.unique()))).mean()) if kind != "numeric" else None}


# ------------------------------------------------------------------ detector
class LeakageDetector:
    def __init__(self, target: str, event_time: str, entity: str | None = None, roles: dict | None = None,
                 birth_columns: list | None = None, cutoff: str | None = None, holdout_fraction: float = 0.3,
                 high_auc: float = 0.99, medium_auc: float = 0.95, instability_gap: float = 0.10,
                 max_rows: int = 600_000, seed: int = 0):
        self.target, self.event_time, self.entity = target, event_time, entity
        self.roles = roles or {}
        self.birth_columns = birth_columns or []
        self.cutoff, self.holdout_fraction = cutoff, holdout_fraction
        self.high_auc, self.medium_auc, self.gap = high_auc, medium_auc, instability_gap
        self.max_rows, self.seed = max_rows, seed

    def _kind(self, df, c):
        if c in self.roles.get("numeric", []) or c in self.roles.get("datetime_epoch", []):
            return "numeric"
        if c in self.roles.get("datetime", []) or pd.api.types.is_datetime64_any_dtype(df[c]):
            return "datetime"
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 50:
            return "numeric"
        return "categorical"

    def detect(self, df: pd.DataFrame, dataset_id: str = "dataset", features: list | None = None) -> LeakageReport:
        t0 = time.time()
        findings: list[Finding] = []
        t = pd.to_datetime(df[self.event_time], errors="coerce")
        cutoff = pd.Timestamp(self.cutoff) if self.cutoff else t.quantile(1 - self.holdout_fraction)
        features = [c for c in (features or df.columns) if c != self.target and not str(c).startswith("_")]
        fit_mask, later_mask = (t < cutoff), (t >= cutoff)
        fit, later = df[fit_mask], df[later_mask]
        if len(fit) > self.max_rows:
            fit = fit.sample(self.max_rows, random_state=self.seed)
        if len(later) > self.max_rows:
            later = later.sample(self.max_rows, random_state=self.seed)
        y_all = df[self.target].astype(int)

        univariate = []
        for c in features:
            kind = self._kind(df, c)

            # 1. names
            if re.search(TARGET_WORDS, c.lower()):
                findings.append(Finding(c, "target_name", "medium", {"name": c},
                                        "column name refers to the target or a fraud outcome",
                                        "confirm the value exists before the prediction is made; otherwise drop"))
            elif re.search(POST_EVENT_WORDS, c.lower()):
                findings.append(Finding(c, "target_name", "low", {"name": c},
                                        "column name suggests information recorded after the transaction",
                                        "confirm it is known at decision time"))

            # 5. post-event timestamps
            if kind == "datetime" and c != self.event_time and c not in self.birth_columns:
                other = pd.to_datetime(df[c], errors="coerce")
                after = float((other > t).mean())
                if after > 0.01:
                    findings.append(Finding(c, "post_event_time", "high", {"share_after_event": round(after, 4)},
                                            f"{after:.1%} of values are later than the event time",
                                            "exclude, or use only values recorded before the event"))

            # 7. target words inside text
            if kind == "categorical" and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])):
                has = df[c].astype("string").str.contains(TARGET_WORDS, case=False, regex=True).fillna(False)
                if has.any():
                    r_pos, r_neg = float(has[y_all == 1].mean()), float(has[y_all == 0].mean())
                    if abs(r_pos - r_neg) > 0.2:
                        findings.append(Finding(c, "target_word_in_text", "high",
                                                {"share_positive": round(r_pos, 4), "share_negative": round(r_neg, 4)},
                                                "target-related words appear in values at very different rates per class",
                                                "remove or mask those words; the text is revealing the label"))
                    elif min(r_pos, r_neg) > 0.9:
                        findings.append(Finding(c, "target_word_in_text", "info",
                                                {"share_positive": round(r_pos, 4), "share_negative": round(r_neg, 4)},
                                                "a target-related word appears in (almost) every value of both classes",
                                                "not a leak, but noise that a language model may misread; strip it"))

            # 2-4. predictive power (datetimes as numbers)
            frame_fit, frame_later = fit, later
            if kind == "datetime":
                conv = lambda s: (pd.to_datetime(s, errors="coerce") - pd.Timestamp("1970-01-01")).dt.total_seconds()
                frame_fit = pd.DataFrame({c: conv(fit[c]), self.target: fit[self.target]})
                frame_later = pd.DataFrame({c: conv(later[c]), self.target: later[self.target]})
            u = univariate_power(frame_fit, frame_later, c, "numeric" if kind == "datetime" else kind, self.target, seed=self.seed)
            u["kind"] = kind
            univariate.append(u)
            a_rand, a_late = u["auc_random_folds"], u["auc_later_period"]
            if np.isnan(a_late):
                continue
            if a_late >= self.high_auc:
                findings.append(Finding(c, "target_copy" if a_rand >= self.high_auc else "suspiciously_predictive", "high",
                                        {"auc_random_folds": round(a_rand, 4), "auc_later_period": round(a_late, 4)},
                                        "one feature alone predicts the target almost perfectly on a later period",
                                        "almost certainly derived from the target or recorded after it; drop unless proven otherwise"))
            elif a_late >= self.medium_auc:
                findings.append(Finding(c, "suspiciously_predictive", "medium",
                                        {"auc_random_folds": round(a_rand, 4), "auc_later_period": round(a_late, 4)},
                                        "unusually strong single-feature predictor",
                                        "verify how and when the value is produced"))
            if kind in ("numeric", "datetime"):
                conv = (lambda x: (pd.to_datetime(x, errors="coerce") - pd.Timestamp("1970-01-01")).dt.total_seconds()) \
                    if kind == "datetime" else (lambda x: pd.to_numeric(x, errors="coerce"))
                fv, lv = conv(fit[c]).dropna(), conv(later[c]).dropna()
                if len(fv) and len(lv):
                    outside = float(((lv < fv.min()) | (lv > fv.max())).mean())
                    if outside >= 0.9:
                        findings.append(Finding(c, "extrapolation_only", "medium",
                                                {"later_values_outside_fit_range": round(outside, 4),
                                                 "auc_random_folds": round(a_rand, 4), "auc_later_period": round(a_late, 4)},
                                                f"{outside:.0%} of later values lie outside the training range (e.g. absolute time): "
                                                "a model can only memorise periods seen in training",
                                                "replace with a cyclical or relative version (hour, day of week, time since event)"))
            if a_rand >= 0.6 and a_rand - a_late >= self.gap:
                findings.append(Finding(c, "temporal_instability", "medium",
                                        {"auc_random_folds": round(a_rand, 4), "auc_later_period": round(a_late, 4),
                                         "unseen_later_values": u.get("later_values_unseen_share")},
                                        "predictive when rows are mixed at random, much weaker on a later period: "
                                        "it memorises specific dates, entities or bursts",
                                        "transform (e.g. hour instead of timestamp) or exclude; use temporal validation"))

        # technical identifiers from the schema
        for c in self.roles.get("row_index", []):
            if c in features:
                findings.append(Finding(c, "technical_identifier", "medium", {"role": "row_index"},
                                        "export row counter: follows file order, which usually follows time",
                                        "exclude from model inputs"))
        for c in self.roles.get("record_id", []):
            if c in features:
                findings.append(Finding(c, "technical_identifier", "info", {"role": "record_id"},
                                        "unique per record: carries no generalisable signal",
                                        "exclude from model inputs; keep for traceability"))

        # 6. identity proxies
        if self.entity and self.entity in df:
            findings.append(Finding(self.entity, "identity_proxy", "medium", {"entities": int(df[self.entity].nunique())},
                                    "entity identifier: lets a model memorise which entities had positives",
                                    "exclude from model inputs; use it only to build history features"))
            cand = [c for c in features if c != self.entity and self._kind(df, c) != "datetime"]
            if cand:
                n_ent = df[self.entity].nunique()
                nun = df.groupby(self.entity)[cand].nunique(dropna=True)
                for c in cand:
                    stable, distinct = float((nun[c] <= 1).mean()), int(df[c].nunique())
                    if stable >= 0.95 and distinct >= 0.2 * n_ent:
                        findings.append(Finding(c, "identity_proxy", "medium",
                                                {"fixed_per_entity": round(stable, 3), "distinct_values": distinct,
                                                 "entities": int(n_ent)},
                                                f"fixed for {stable:.0%} of entities with {distinct} distinct values: stands in for the entity",
                                                "exclude raw values; a coarse transformation (e.g. decile) may be acceptable"))

            # 8. cold start
            seen = set(df.loc[fit_mask, self.entity])
            later_all = df[later_mask]
            unseen = ~later_all[self.entity].isin(seen)
            if unseen.any() and (~unseen).any():
                r_new = float(later_all.loc[unseen, self.target].mean())
                r_old = float(later_all.loc[~unseen, self.target].mean())
                share_pos = float(later_all.loc[unseen, self.target].sum() / max(later_all[self.target].sum(), 1))
                ratio = r_new / r_old if r_old > 0 else float("inf")
                ev = {"later_rows_new_entities": int(unseen.sum()), "target_rate_new_entities": round(r_new, 4),
                      "target_rate_known_entities": round(r_old, 6), "rate_ratio": round(ratio, 1),
                      "share_of_later_positives_from_new_entities": round(share_pos, 4)}
                if ratio >= 10 or r_new >= 0.5:
                    findings.append(Finding(f"<history of {self.entity}>", "cold_start_artifact", "high", ev,
                                            "entities never seen before are almost always positive: any feature encoding "
                                            "history length, first appearance or 'no previous transactions' becomes a label proxy",
                                            "report results separately for new and known entities; test history features with and without this slice"))
                elif ratio >= 3:
                    findings.append(Finding(f"<history of {self.entity}>", "cold_start_artifact", "low", ev,
                                            "new entities have a clearly higher target rate", "report new vs known entities separately"))

            # 9. concentration
            pos_by_entity = df.groupby(self.entity)[self.target].sum().sort_values(ascending=False)
            if pos_by_entity.sum() > 0:
                top = max(1, int(0.05 * len(pos_by_entity)))
                findings.append(Finding(self.entity, "entity_concentration", "info",
                                        {"entities_with_positive": int((pos_by_entity > 0).sum()),
                                         "share_of_positives_in_top_5pct_entities": round(float(pos_by_entity.iloc[:top].sum() / pos_by_entity.sum()), 4)},
                                        "how concentrated positives are across entities",
                                        "high concentration makes entity memorisation more tempting; keep temporal validation"))

        return LeakageReport(dataset_id=dataset_id, target=self.target, created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                             seconds=round(time.time() - t0, 1),
                             split={"cutoff": str(cutoff), "fit_rows": int(fit_mask.sum()), "holdout_rows": int(later_mask.sum())},
                             findings=findings, univariate=sorted(univariate, key=lambda u: -u["auc_later_period"]
                                                                  if not np.isnan(u["auc_later_period"]) else 0))


def check_point_in_time(df: pd.DataFrame, feature_fn, time_col: str, group_col: str, n_samples: int = 200,
                        seed: int = 0, atol: float = 1e-6) -> dict:
    """Recompute features for sampled rows using only that row's group history up to (and including) the row.

    feature_fn(frame) must return a DataFrame of feature columns indexed like `frame`. If a value computed on
    the full data differs from the value computed on the truncated history, the feature used future rows.
    """
    order = df.sort_values([group_col, time_col], kind="mergesort")
    full = feature_fn(order)
    picks = order.sample(min(n_samples, len(order)), random_state=seed).index
    pos = pd.Series(np.arange(len(order)), index=order.index)
    mismatches = {c: 0 for c in full.columns}
    examples = []
    for idx in picks:
        g = order[group_col].loc[idx]
        group = order[order[group_col] == g]
        history = group[pos.loc[group.index] <= pos.loc[idx]]
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
                    examples.append({"row": str(idx), "feature": c, "full_data_value": a, "history_only_value": b})
    leaking = [c for c, n in mismatches.items() if n]
    return {"rows_checked": int(len(picks)), "mismatches": mismatches, "leaking_features": leaking,
            "passed": not leaking, "examples": examples}
