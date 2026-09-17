"""Built-in sanity transformer: a deliberately small PyTorch transformer encoder for tabular transactions.

Purpose: verify that a prepared dataset is valid, consistent and learnable end to end (it is not meant to
compete with large language models). Runs on CPU for demo-sized data.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.metrics import classification_metrics
from src.models.base import ModelAdapter, basic_data_checks, check
from src.representation.tabular_tokenizer import TabularTokenizer


def sinusoidal_positions(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class SanityTransformerModel(nn.Module):
    """token embedding + feature-position embedding + sinusoidal positional encoding -> encoder -> pooling -> head"""

    def __init__(self, vocab_size: int, n_positions: int, d_model=64, n_heads=4, n_layers=2, dim_feedforward=128,
                 dropout=0.1, pooling="cls"):
        super().__init__()
        self.pooling = pooling
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.feature_position = nn.Embedding(n_positions, d_model)
        self.register_buffer("positional", sinusoidal_positions(n_positions, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        h = self.token(ids) + self.feature_position(positions) + self.positional[: ids.shape[1]]
        h = self.encoder(h)
        pooled = h[:, 0] if self.pooling == "cls" else h[:, 1:].mean(dim=1)
        return self.head(self.norm(pooled)).squeeze(-1)


class SanityTransformerAdapter(ModelAdapter):
    key = "sanity_transformer"
    label = "Built-in Sanity Transformer"
    description = "Small PyTorch transformer encoder; validates that prepared data is learnable. CPU-friendly."

    def __init__(self, config: dict, device: str | None = None):
        super().__init__(config)
        self.mcfg = config["models"]["sanity_transformer"]
        self.rcfg = config["representation"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: TabularTokenizer | None = None
        self.model: SanityTransformerModel | None = None

    def describe(self) -> dict:
        n = sum(p.numel() for p in self.model.parameters()) if self.model else None
        return {"model": self.label, "device": self.device, "parameters": n,
                **{k: self.mcfg[k] for k in ["d_model", "n_heads", "n_layers", "pooling"]}}

    def _build(self, prepared):
        self.tokenizer = TabularTokenizer(self.rcfg["numeric_bins"], self.rcfg["min_category_count"]).fit(
            prepared.frames["train"], prepared.numeric, prepared.categorical)
        m = self.mcfg
        self.model = SanityTransformerModel(self.tokenizer.vocab_size, self.tokenizer.n_positions, m["d_model"], m["n_heads"],
                                            m["n_layers"], m["dim_feedforward"], m["dropout"], m["pooling"]).to(self.device)

    @torch.no_grad()
    def predict(self, frame: pd.DataFrame, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        ids = torch.from_numpy(self.tokenizer.transform(frame))
        out = []
        for i in range(0, len(ids), batch_size):
            out.append(torch.sigmoid(self.model(ids[i:i + batch_size].to(self.device))).double().cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def train(self, prepared, settings: dict, progress=None) -> dict:
        s = {**self.mcfg, **(settings or {})}
        seed = int(s.get("seed", 42))
        # Single-threaded: multi-threaded CPU reductions are not bit-reproducible in PyTorch, and this model is
        # small enough that the speed cost is negligible. Without this, the same seed can give different runs
        # (observed directly: identical settings gave val PR-AUC 0.91 in one process and 0.10 in another).
        torch.set_num_threads(1)
        torch.manual_seed(seed); np.random.seed(seed)
        self._build(prepared)
        train, val = prepared.frames["train"], prepared.frames["validation"]
        x = torch.from_numpy(self.tokenizer.transform(train))
        y = torch.tensor(train["_target"].to_numpy(dtype="float32"))
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(s["learning_rate"]), weight_decay=0.01)
        # Audit finding 4: the sampler oversamples the positive class (sampling.train_positive_share, e.g. 10%
        # against a true ~0.6% rate) and computes an inverse-probability `_weight` per row to correct for this
        # — but that weight was only ever used at evaluation time, never in the training loss, a real
        # train/eval distribution mismatch. class_weighting="sample_weight" (reusing `_weight` as a per-example
        # loss weight) was implemented as the fix, but MEASURED afterward (3 seeds, synthetic data) to cause a
        # large, consistent PR-AUC regression at severe imbalance (confirmed in isolation on E0 alone:
        # 0.47 -> 0.14) — reweighting toward the true rate makes the rare class's gradient signal negligible,
        # undermining what oversampling exists to provide. "none" (the default, and the original behaviour) is
        # empirically better on this data; "sample_weight" is kept available for direct comparison, not as a
        # recommended setting.
        class_weighting = s.get("class_weighting", "none")
        if class_weighting not in ("none", "sample_weight"):
            raise ValueError(f"models.sanity_transformer.class_weighting must be 'none' or 'sample_weight', got {class_weighting!r}")
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        sample_weight = (torch.tensor(train["_weight"].to_numpy(dtype="float32"))
                         if class_weighting == "sample_weight" and "_weight" in train else None)
        g = torch.Generator().manual_seed(seed)
        history, best, best_state, bad_epochs = [], -1.0, None, 0
        t0 = time.time()
        for epoch in range(int(s["epochs"])):
            self.model.train()
            perm = torch.randperm(len(x), generator=g)
            total, n = 0.0, 0
            for i in range(0, len(x), int(s["batch_size"])):
                idx = perm[i:i + int(s["batch_size"])]
                logits = self.model(x[idx].to(self.device))
                losses = loss_fn(logits, y[idx].to(self.device))
                if sample_weight is not None:
                    w = sample_weight[idx].to(self.device)
                    loss = (losses * w).sum() / w.sum()
                else:
                    loss = losses.mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx); n += len(idx)
            pv = self.predict(val)
            vm = classification_metrics(val["_target"], pv, val["_weight"], 0.5)
            entry = {"epoch": epoch + 1, "train_loss": total / max(n, 1), "val_pr_auc": vm["pr_auc"],
                     "val_log_loss": vm["log_loss"], "seconds": round(time.time() - t0, 2)}
            history.append(entry)
            if progress:
                progress(epoch + 1, int(s["epochs"]), entry)
            score = vm["pr_auc"] if vm["pr_auc"] is not None else -vm["log_loss"]
            if score > best:
                best, bad_epochs = score, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= int(s["patience"]):
                    break
        if best_state:
            self.model.load_state_dict(best_state)
        return {"history": history, "best_val_pr_auc": best, "epochs_run": len(history),
                "train_seconds": round(time.time() - t0, 2), "device": self.device, "settings": s}

    def sanity_checks(self, prepared, max_rows: int = 5000) -> list[dict]:
        """Runs the full pipeline (encode -> forward pass -> a short training run -> predict) as a diagnostic.
        The short training run and its tiny sample are throwaway: this must not leave a real, already-trained
        adapter (self.model / self.tokenizer) worse off than before the call, so the original state — if any —
        is restored afterwards regardless of how the checks finish."""
        checks = basic_data_checks(prepared)
        if any(c["status"] == "fail" for c in checks):
            return checks
        saved_model, saved_tokenizer = self.model, self.tokenizer
        try:
            return checks + self._run_pipeline_checks(prepared, max_rows)
        finally:
            self.model, self.tokenizer = saved_model, saved_tokenizer

    def _run_pipeline_checks(self, prepared, max_rows: int) -> list[dict]:
        checks = []
        try:
            self._build(prepared)
            enc = {sp: self.tokenizer.transform(f) for sp, f in prepared.frames.items()}
            checks.append(check("Encoding successful", True, f"vocabulary {self.tokenizer.vocab_size:,} tokens"))
        except Exception as e:
            return checks + [check("Encoding successful", False, str(e))]
        dims = {sp: e.shape[1] for sp, e in enc.items()}
        checks.append(check("Feature dimensions consistent", len(set(dims.values())) == 1, f"sequence length per split: {dims}"))
        src_nan = int(sum(pd.to_numeric(f[c], errors="coerce").isna().sum() for f in prepared.frames.values() for c in prepared.numeric))
        all_ids = np.concatenate([e.ravel() for e in enc.values()])
        checks.append(check("No NaN", True, f"model inputs are token ids; {src_nan:,} missing numeric values mapped to missing tokens"))
        inf = int(sum(np.isinf(pd.to_numeric(f[c], errors="coerce")).sum() for f in prepared.frames.values() for c in prepared.numeric))
        checks.append(check("No Inf", inf == 0 or None, f"{inf} infinite values" + (" mapped to missing tokens" if inf else "")))
        checks.append(check("Token ids in range", bool(all_ids.min() >= 0 and all_ids.max() < self.tokenizer.vocab_size)))
        try:
            with torch.no_grad():
                out = self.model(torch.from_numpy(enc["train"][:64]).to(self.device))
            checks.append(check("Forward pass successful", bool(torch.isfinite(out).all()), f"output shape {tuple(out.shape)}"))
        except Exception as e:
            return checks + [check("Forward pass successful", False, str(e))]
        small = prepared.__class__(prepared.level, {sp: f.sample(min(len(f), max_rows), random_state=0) if sp == "train" else f
                                                    for sp, f in prepared.frames.items()}, prepared.numeric, prepared.categorical, prepared.info)
        try:
            summary = self.train(small, {"epochs": 2, "patience": 5})
            losses = [h["train_loss"] for h in summary["history"]]
            checks.append(check("Training completed", all(np.isfinite(losses)), f"losses per epoch {np.round(losses, 4).tolist()}"))
            checks.append(check("Loss decreased", losses[-1] < losses[0] or None, "loss did not decrease in 2 epochs" if losses[-1] >= losses[0] else ""))
        except Exception as e:
            return checks + [check("Training completed", False, str(e))]
        p = self.predict(prepared.frames["test"])
        checks.append(check("Predictions generated", bool(len(p) == len(prepared.frames["test"]) and np.isfinite(p).all()
                                                          and p.min() >= 0 and p.max() <= 1),
                            f"{len(p):,} probabilities, range {p.min():.3f}–{p.max():.3f}"))
        checks.append(check("Predictions vary", bool(np.std(p) > 1e-6) or None, f"std {np.std(p):.4f}"))
        return checks

    def save(self, path) -> Path:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "tokenizer.json").write_text(json.dumps(self.tokenizer.to_dict()))
        (path / "adapter.json").write_text(json.dumps({"key": self.key, "model_config": self.mcfg, "threshold": self.threshold}))
        return path

    def load(self, path) -> "SanityTransformerAdapter":
        path = Path(path)
        meta = json.loads((path / "adapter.json").read_text())
        self.mcfg, self.threshold = meta["model_config"], meta["threshold"]
        self.tokenizer = TabularTokenizer.from_dict(json.loads((path / "tokenizer.json").read_text()))
        m = self.mcfg
        self.model = SanityTransformerModel(self.tokenizer.vocab_size, self.tokenizer.n_positions, m["d_model"], m["n_heads"],
                                            m["n_layers"], m["dim_feedforward"], m["dropout"], m["pooling"]).to(self.device)
        self.model.load_state_dict(torch.load(path / "model.pt", map_location=self.device))
        return self
