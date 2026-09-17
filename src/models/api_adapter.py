"""Custom / API model: inference-only backend that scores text rows through an HTTP endpoint.

Expected contract: POST {"prompts": [...]} -> {"probabilities": [...]} (same length, values in [0, 1]).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ModelAdapter, basic_data_checks, check
from src.representation.text_builder import prompts, spec_for


class APIAdapter(ModelAdapter):
    key = "api"
    label = "Custom/API Model"
    description = "Scores rows through your own HTTP endpoint. Inference only: training happens on the API side."

    def __init__(self, config: dict):
        super().__init__(config)
        self.acfg = config["models"]["api"]
        self.spec = None

    @classmethod
    def availability(cls, hardware: dict, config: dict) -> tuple[bool, str]:
        endpoint = config["models"]["api"].get("endpoint")
        return (True, f"endpoint {endpoint}") if endpoint else (False, "no endpoint configured (models.api.endpoint in config.yaml)")

    def train(self, prepared, settings: dict, progress=None) -> dict:
        self.spec = spec_for(prepared)
        return {"note": "API backend is inference only; no local training performed", "train_seconds": 0.0, "history": []}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        import requests
        if self.spec is None:
            raise RuntimeError("call train() first so the text representation is fixed")
        texts, out = prompts(frame, self.spec).tolist(), []
        for i in range(0, len(texts), int(self.acfg["batch_size"])):
            r = requests.post(self.acfg["endpoint"], json={"prompts": texts[i:i + int(self.acfg["batch_size"])]},
                              timeout=self.acfg["timeout_seconds"])
            r.raise_for_status()
            probs = r.json().get("probabilities")
            if not isinstance(probs, list) or len(probs) != len(texts[i:i + int(self.acfg["batch_size"])]):
                raise ValueError("API response must contain 'probabilities' with one value per prompt")
            out.extend(probs)
        arr = np.asarray(out, dtype=float)
        if not (np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1):
            raise ValueError("API returned probabilities outside [0, 1]")
        return arr

    def sanity_checks(self, prepared) -> list[dict]:
        checks = basic_data_checks(prepared)
        checks.append(check("Endpoint configured", bool(self.acfg.get("endpoint")), self.acfg.get("endpoint") or "not set"))
        return checks

    def save(self, path) -> Path:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        (path / "adapter.json").write_text(json.dumps({"key": self.key, "spec": self.spec.to_dict() if self.spec else None,
                                                       "threshold": self.threshold, "endpoint": self.acfg.get("endpoint")}))
        return path

    def load(self, path) -> "APIAdapter":
        from src.representation.transaction_formatter import RepresentationSpec
        meta = json.loads((Path(path) / "adapter.json").read_text())
        self.spec = RepresentationSpec.from_dict(meta["spec"]) if meta["spec"] else None
        self.threshold = meta["threshold"]
        return self
