"""Diagnostics for detecting information loss between E0/E1/E2 before expensive training.

These checks are deliberately model-light.  They answer questions such as:
- did categorical preprocessing collapse many validation/test values to UNKNOWN?
- did quantile tokenisation compress a high-cardinality numeric feature too aggressively?
- did a processed level remove simple target-associated signal that existed in another level?
- did feature engineering add many near-constant/noisy positions?

All mappings/statistics are fitted on the training split only.  Test labels are used only for diagnostic
measurement, never to construct a feature representation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.representation.tabular_tokenizer import TabularTokenizer


def _safe_auc(y, score, weight=None):
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    good = np.isfinite(score) & pd.notna(y)
    if good.sum() < 10 or len(np.unique(y[good])) < 2:
        return None
    w = np.asarray(weight)[good] if weight is not None else None
    try:
        auc = float(roc_auc_score(y[good], score[good], sample_weight=w))
    except ValueError:
        return None
    # Univariate signal strength should not depend on whether larger values imply positive or negative risk.
    return max(auc, 1.0 - auc)


def _numeric_auc(train: pd.DataFrame, test: pd.DataFrame, feature: str):
    x = pd.to_numeric(test[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).median()
    x = x.fillna(0.0 if pd.isna(median) else float(median))
    return _safe_auc(test["_target"], x, test.get("_weight"))


def _categorical_auc(train: pd.DataFrame, test: pd.DataFrame, feature: str):
    # Leakage-safe target-rate encoding for diagnostics: mapping is learned on train and only scored on test.
    tr = train[[feature, "_target"]].copy()
    tr[feature] = tr[feature].astype("string").fillna("<MISSING>")
    global_rate = float(tr["_target"].mean())
    rates = tr.groupby(feature, observed=True)["_target"].mean()
    te = test[feature].astype("string").fillna("<MISSING>")
    score = te.map(rates).fillna(global_rate).to_numpy(dtype=float)
    return _safe_auc(test["_target"], score, test.get("_weight"))


def diagnose_level(prepared, representation_cfg: dict) -> dict:
    """Return compact representation + signal diagnostics for one PreparedLevel."""
    train = prepared.frames["train"]
    test = prepared.frames["test"]
    # Very rare fraud data can yield a later temporal split with no positives at small sample sizes.
    # Prefer test, but fall back to validation for label-based diagnostics rather than fabricate an AUC.
    eval_frame = test if test["_target"].nunique(dropna=True) >= 2 else prepared.frames["validation"]
    eval_split = "test" if eval_frame is test else "validation"
    tok = TabularTokenizer(
        representation_cfg.get("numeric_bins", 16),
        representation_cfg.get("min_category_count", 5),
        numeric_mode=representation_cfg.get("numeric_mode", "quantile_bin"),
        coarse_bins=representation_cfg.get("numeric_coarse_bins", 0),
        numeric_clip=representation_cfg.get("numeric_clip"),
        continuous_features=representation_cfg.get("continuous_features", []),
    ).fit(train, prepared.numeric, prepared.categorical)

    test_ids = tok.transform(eval_frame)
    unique_rows = len(np.unique(test_ids, axis=0)) if len(test_ids) else 0

    features = []
    for c in prepared.numeric:
        raw_unique = int(pd.to_numeric(train[c], errors="coerce").nunique(dropna=True))
        if tok._is_continuous(c) and tok.coarse_bins == 0:
            represented_unique = raw_unique
            retention = 1.0 if raw_unique else None
            unknown_rate = None
        else:
            j = 1 + prepared.numeric.index(c)
            represented_unique = int(np.unique(tok.transform(train)[:, j]).size)
            retention = float(represented_unique / raw_unique) if raw_unique else None
            unknown_rate = None
        features.append({
            "feature": c, "kind": "numeric", "train_unique": raw_unique,
            "represented_unique": represented_unique, "representation_retention": retention,
            "test_missing_rate": float(pd.to_numeric(eval_frame[c], errors="coerce").isna().mean()),
            "test_unknown_rate": unknown_rate, "univariate_test_auc": _numeric_auc(train, eval_frame, c),
        })

    cat_offset = 1 + len(prepared.numeric)
    train_ids = tok.transform(train)
    for k, c in enumerate(prepared.categorical):
        j = cat_offset + k
        unknown_id = tok.special[c]["unknown"]
        missing_id = tok.special[c]["missing"]
        ids = test_ids[:, j] if len(test_ids) else np.array([], dtype=int)
        non_missing = ids != missing_id
        unknown_rate = float((ids[non_missing] == unknown_id).mean()) if non_missing.any() else 0.0
        raw_unique = int(train[c].astype("string").nunique(dropna=True))
        represented_unique = int(np.unique(train_ids[:, j]).size) if len(train_ids) else 0
        features.append({
            "feature": c, "kind": "categorical", "train_unique": raw_unique,
            "represented_unique": represented_unique,
            "representation_retention": float(represented_unique / raw_unique) if raw_unique else None,
            "test_missing_rate": float(eval_frame[c].isna().mean()), "test_unknown_rate": unknown_rate,
            "univariate_test_auc": _categorical_auc(train, eval_frame, c),
        })

    aucs = [f["univariate_test_auc"] for f in features if f["univariate_test_auc"] is not None]
    unknowns = [f["test_unknown_rate"] for f in features if f["test_unknown_rate"] is not None]
    retained = [f["representation_retention"] for f in features if f["representation_retention"] is not None]
    near_constant = sum(f["represented_unique"] <= 2 for f in features)
    return {
        "level": prepared.level,
        "n_features": len(features),
        "numeric_features": len(prepared.numeric),
        "categorical_features": len(prepared.categorical),
        "evaluation_split": eval_split,
        "test_rows": len(eval_frame),
        "unique_token_row_rate": float(unique_rows / len(test_ids)) if len(test_ids) else None,
        "near_constant_representations": int(near_constant),
        "mean_representation_retention": float(np.mean(retained)) if retained else None,
        "mean_test_unknown_rate": float(np.mean(unknowns)) if unknowns else None,
        "best_univariate_test_auc": float(max(aucs)) if aucs else None,
        "median_univariate_test_auc": float(np.median(aucs)) if aucs else None,
        "features": features,
    }


def compare_levels(levels: dict, representation_cfg: dict) -> pd.DataFrame:
    """One-row-per-level summary, convenient for notebooks and automated audit reports."""
    rows = []
    for name, prepared in levels.items():
        d = diagnose_level(prepared, representation_cfg)
        rows.append({k: v for k, v in d.items() if k != "features"})
    return pd.DataFrame(rows).sort_values("level").reset_index(drop=True)
