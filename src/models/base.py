"""Model interface. Every backend (built-in transformer, Hugging Face, Nemotron, API) implements this contract,
so the data pipeline never depends on a particular model.

`train`, `predict`, `save` and `load` are backend-specific. `evaluate` and `sanity_checks` have a shared,
correct default here (threshold chosen on validation by best weighted F1, then applied unchanged to test) so
every backend reports metrics the same way; a backend only needs to override them if it has something extra
to check.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import best_f1_threshold, classification_metrics


class ModelUnavailable(RuntimeError):
    """Raised when a backend cannot run on the detected hardware or with the current configuration."""


class ModelAdapter(ABC):
    key: str = "base"
    label: str = "Base model"
    description: str = ""

    def __init__(self, config: dict):
        self.config = config
        self.threshold: float | None = None

    # ---- capability -------------------------------------------------------------------
    @classmethod
    def availability(cls, hardware: dict, config: dict) -> tuple[bool, str]:
        """Whether this backend can run on the detected hardware, with a message for the UI."""
        return True, "available"

    def describe(self) -> dict:
        return {"model": self.label}

    # ---- lifecycle ----------------------------------------------------------------------
    @abstractmethod
    def train(self, prepared, settings: dict, progress=None) -> dict:
        """Fit on prepared.frames['train'], using 'validation' for model selection. Returns a training summary."""

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class for each row of a prepared frame."""

    @abstractmethod
    def save(self, path: str | Path) -> Path: ...

    @abstractmethod
    def load(self, path: str | Path) -> "ModelAdapter": ...

    # ---- shared evaluation ----------------------------------------------------------------
    def evaluate(self, prepared) -> dict:
        """Threshold chosen on validation (best weighted F1), then applied unchanged to test."""
        val, test = prepared.frames["validation"], prepared.frames["test"]
        pv = self.predict(val)
        self.threshold = best_f1_threshold(val["_target"], pv, val["_weight"]) if val["_target"].nunique() == 2 else 0.5
        pt = self.predict(test)
        out = {"threshold_from_validation": self.threshold,
               "validation": classification_metrics(val["_target"], pv, val["_weight"], self.threshold),
               "test": classification_metrics(test["_target"], pt, test["_weight"], self.threshold),
               "test_unweighted": classification_metrics(test["_target"], pt, None, self.threshold)}
        known = ~test["_new_entity"].astype(bool)
        if known.any() and (~known).any() and test.loc[known, "_target"].nunique() == 2:
            out["test_known_entities"] = classification_metrics(test.loc[known, "_target"], pt[known.to_numpy()],
                                                                test.loc[known, "_weight"], self.threshold)
        return out, pv, pt

    def sanity_checks(self, prepared) -> list[dict]:
        return basic_data_checks(prepared)


def check(name: str, ok: bool | None, detail: str = "") -> dict:
    return {"check": name, "status": "pass" if ok else ("warn" if ok is None else "fail"), "detail": detail}


def basic_data_checks(prepared) -> list[dict]:
    fr = prepared.frames
    splits_ok = all(len(fr[s]) for s in ["train", "validation", "test"])
    y = fr["train"]["_target"]
    return [
        check("Schema valid", bool(prepared.features) and splits_ok,
              f"{len(prepared.features)} features; rows " + ", ".join(f"{s}={len(f):,}" for s, f in fr.items())),
        check("Labels valid", set(pd.concat([f["_target"] for f in fr.values()]).dropna().unique()) <= {0, 1} and y.nunique() == 2,
              f"training positives {int(y.sum()):,} of {len(y):,}"),
    ]


# Backends planned for the UI; status is shown honestly until each is implemented.
PLANNED_BACKENDS = [
    {"key": "sanity_transformer", "label": "Built-in Sanity Transformer", "phase": 5,
     "description": "Small tabular transformer trained from scratch; runs on CPU.", "implemented": True},
    {"key": "huggingface", "label": "Hugging Face Model", "phase": 7,
     "description": "Configurable model from the Hugging Face Hub through an adapter.", "implemented": False},
    {"key": "nemotron", "label": "Nemotron", "phase": 7,
     "description": "Small NVIDIA Nemotron checkpoint (configurable); requires a CUDA GPU.", "implemented": False},
    {"key": "api", "label": "Custom/API Model", "phase": 7,
     "description": "Your own model behind an HTTP endpoint.", "implemented": False},
]
