"""Metrics for imbalanced binary classification, weighted to the real class balance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as M


def classification_metrics(y, p, w=None, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    w = None if w is None else np.asarray(w, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {"threshold": float(threshold), "n": int(len(y)), "positives": int(y.sum())}
    out["accuracy"] = float(M.accuracy_score(y, pred, sample_weight=w))
    out["precision"] = float(M.precision_score(y, pred, sample_weight=w, zero_division=0))
    out["recall"] = float(M.recall_score(y, pred, sample_weight=w, zero_division=0))
    out["f1"] = float(M.f1_score(y, pred, sample_weight=w, zero_division=0))
    both = len(np.unique(y)) == 2
    out["roc_auc"] = float(M.roc_auc_score(y, p, sample_weight=w)) if both else None
    out["pr_auc"] = float(M.average_precision_score(y, p, sample_weight=w)) if both else None
    out["log_loss"] = float(M.log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), sample_weight=w, labels=[0, 1]))
    tn, fp, fn, tp = M.confusion_matrix(y, pred, labels=[0, 1], sample_weight=w).ravel()
    out["confusion_matrix"] = {"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)}
    return out


def best_f1_threshold(y, p, w=None, grid: int = 200) -> float:
    """Threshold maximising (weighted) F1; chosen on validation, then applied unchanged to test."""
    p = np.asarray(p, dtype=float)
    candidates = np.unique(np.quantile(p, np.linspace(0, 1, grid)))
    scores = [M.f1_score(y, (p >= t).astype(int), sample_weight=w, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate(pred: pd.DataFrame, threshold: float) -> dict:
    """pred columns: label, prob, _weight (optional), _slice_unseen_entity (optional)."""
    w = pred["_weight"] if "_weight" in pred else None
    out = {"weighted": classification_metrics(pred["label"], pred["prob"], w, threshold) if w is not None else None,
           "unweighted": classification_metrics(pred["label"], pred["prob"], None, threshold),
           "weighted_at_0_5": classification_metrics(pred["label"], pred["prob"], w, 0.5) if w is not None else None}
    if "_slice_unseen_entity" in pred and pred["_slice_unseen_entity"].any():
        s = pred[pred["_slice_unseen_entity"]]
        k = pred[~pred["_slice_unseen_entity"] & (pred["label"] == 1)]
        out["slices"] = {
            "unseen_entity": {"rows": int(len(s)), "positives": int(s["label"].sum()),
                              "recall": float(((s["prob"] >= threshold) & (s["label"] == 1)).sum() / max(s["label"].sum(), 1)),
                              "mean_prob": float(s["prob"].mean())},
            "known_entity_positives": {"rows": int(len(k)),
                                       "recall": float((k["prob"] >= threshold).mean()) if len(k) else None},
        }
    return out


