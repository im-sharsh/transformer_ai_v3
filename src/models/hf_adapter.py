"""Hugging Face causal language models (and Nemotron) through the common adapter interface."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ModelAdapter, ModelUnavailable, basic_data_checks, check
from src.representation.text_builder import prompts, spec_for


def _libraries_ok() -> tuple[bool, str]:
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
        return True, ""
    except ImportError as e:
        return False, f"missing library: {e.name} (pip install transformers peft accelerate)"


class HuggingFaceAdapter(ModelAdapter):
    key = "huggingface"
    label = "Hugging Face Model"
    description = "Any compatible causal language model from the Hugging Face Hub, fine-tuned with LoRA on text rows."
    section = "huggingface"

    def __init__(self, config: dict):
        super().__init__(config)
        self.mcfg = dict(config["models"][self.section])
        self.model = self.tokenizer = self.label_ids = self.spec = None

    @classmethod
    def availability(cls, hardware: dict, config: dict) -> tuple[bool, str]:
        ok, msg = _libraries_ok()
        if not ok:
            return False, msg
        if not hardware["cuda"]:
            return True, "CPU only: fine-tuning even a small language model will be very slow; use demo-sized data"
        return True, f"GPU {hardware['gpu_name']} available"

    def describe(self) -> dict:
        from src.models.hf_backend import count_parameters, pick_device_and_dtype
        device, dtype = pick_device_and_dtype()
        return {"model": self.label, "model_id": self.mcfg["model_id"], "device": device, "dtype": str(dtype),
                **(count_parameters(self.model) if self.model is not None else {})}

    def _load(self):
        from src.models import hf_backend as B
        device, dtype = B.pick_device_and_dtype()
        kwargs = {"model_id": self.mcfg["model_id"], "load_in_4bit": bool(self.mcfg.get("load_in_4bit")) and device == "cuda",
                 "lora_r": self.mcfg["lora_r"], "lora_alpha": self.mcfg["lora_alpha"], "lora_dropout": self.mcfg["lora_dropout"],
                 "gradient_checkpointing": device == "cuda"}
        if self.mcfg.get("target_modules"):          # LoRA target module names are architecture-specific (e.g.
            kwargs["target_modules"] = list(self.mcfg["target_modules"])   # q_proj/... for LLaMA, c_attn for GPT-2)
        mc = B.ModelConfig(**kwargs)
        self.tokenizer = B.load_tokenizer(mc.model_id)
        base = B.load_base_model(mc, dtype)
        if device == "cuda" and not mc.load_in_4bit:
            base = base.to("cuda")
        self.model = B.attach_lora(base, mc)
        return device, dtype

    def train(self, prepared, settings: dict, progress=None) -> dict:
        from src.models import hf_backend as B
        s = {**self.mcfg, **(settings or {})}
        device, dtype = self._load()
        self.spec = spec_for(prepared)
        tr, va = prepared.frames["train"], prepared.frames["validation"]
        p_tr, p_va = prompts(tr, self.spec), prompts(va, self.spec)
        self.label_ids = B.label_token_ids(self.tokenizer, p_tr.iloc[0], self.spec.label_texts)
        max_tokens = int(self.config["representation"]["max_tokens"])
        tc = B.TrainConfig(epochs=int(s["epochs"]), batch_size=int(s["batch_size"]),
                           gradient_accumulation_steps=int(s["gradient_accumulation_steps"]),
                           learning_rate=float(s["learning_rate"]), max_tokens=max_tokens, seed=int(s.get("seed", 42)),
                           eval_every_steps=int(s.get("eval_every_steps", 100)), checkpoint_every_steps=int(s.get("checkpoint_every_steps", 100)))
        out_dir = Path(s.get("run_dir", Path(self.config["project"]["experiments_dir"]) / "_checkpoints" / self.key))
        trainer = B.Trainer(self.model, self.tokenizer, self.label_ids, tc, out_dir, device, dtype)
        summary = trainer.fit(p_tr, tr["_target"].astype(int), p_va, va["_target"].astype(int), va["_weight"].to_numpy())
        best = out_dir / "best_model" / "adapter.pt"
        if best.exists():
            import torch
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.model, torch.load(best, weights_only=False))
        self._device, self._dtype = device, dtype
        return {**{k: v for k, v in summary.items() if k != "history"}, "history": summary["history"], "device": device, "settings": s}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        from src.models import hf_backend as B
        enc = B.Encoded(self.tokenizer, prompts(frame, self.spec), frame["_target"].astype(int), int(self.config["representation"]["max_tokens"]))
        device, dtype = B.pick_device_and_dtype()
        return B.predict(self.model, enc, self.label_ids, device, dtype, batch_size=32)

    def sanity_checks(self, prepared) -> list[dict]:
        checks = basic_data_checks(prepared)
        spec = spec_for(prepared)
        text = prompts(prepared.frames["train"].head(3), spec)
        checks.append(check("Text representation built", len(text) > 0, text.iloc[0][:160] + " …" if len(text) else ""))
        ok, msg = _libraries_ok()
        checks.append(check("Libraries installed", ok, msg))
        return checks

    def save(self, path) -> Path:
        import torch
        from peft import get_peft_model_state_dict
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        torch.save(get_peft_model_state_dict(self.model), path / "lora_adapter.pt")
        (path / "adapter.json").write_text(json.dumps({"key": self.key, "model_config": self.mcfg, "threshold": self.threshold,
                                                       "spec": self.spec.to_dict(), "label_ids": self.label_ids}))
        return path

    def load(self, path) -> "HuggingFaceAdapter":
        import torch
        from peft import set_peft_model_state_dict
        from src.representation.transaction_formatter import RepresentationSpec
        path = Path(path)
        meta = json.loads((path / "adapter.json").read_text())
        self.mcfg, self.threshold = meta["model_config"], meta["threshold"]
        self._load()
        set_peft_model_state_dict(self.model, torch.load(path / "lora_adapter.pt", weights_only=False))
        self.spec = RepresentationSpec.from_dict(meta["spec"])
        self.label_ids = {int(k): v for k, v in meta["label_ids"].items()}
        return self


class NemotronAdapter(HuggingFaceAdapter):
    key = "nemotron"
    label = "Nemotron"
    description = "NVIDIA Nemotron small checkpoint (configurable), 4-bit QLoRA fine-tuning. Requires a CUDA GPU."
    section = "nemotron"

    @classmethod
    def availability(cls, hardware: dict, config: dict) -> tuple[bool, str]:
        ok, msg = _libraries_ok()
        if not ok:
            return False, msg
        need = config["models"]["nemotron"].get("min_gpu_memory_gb", 12)
        if not hardware["cuda"]:
            return False, "requires a CUDA GPU (e.g. Colab T4); not practical on CPU"
        if (hardware["gpu_memory_gb"] or 0) < need:
            return False, f"needs at least {need} GB GPU memory; found {hardware['gpu_memory_gb']} GB"
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            return False, "missing library: bitsandbytes (pip install bitsandbytes)"
        return True, f"{config['models']['nemotron']['model_id']} on {hardware['gpu_name']}"
