"""Causal language model backend (Hugging Face / Nemotron): label-token scoring, LoRA, resumable training.

The prompt ends with 'Answer:'. The logits of the two label tokens (' no' / ' yes') at the last
position form a two-class output; only those two rows of the output layer are computed.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.metrics import classification_metrics

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class ModelConfig:
    model_id: str = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: list(DEFAULT_TARGETS))
    gradient_checkpointing: bool = True

    def to_dict(self):
        return asdict(self)


def pick_device_and_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return "cuda", (torch.bfloat16 if major >= 8 else torch.float16)
    return "cpu", torch.float32


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"           # the last position is always the answer slot
    return tok


def label_token_ids(tokenizer, example_prompt: str, label_texts: dict) -> dict:
    """Find the single token each label adds after a real prompt (robust to tokenizer merges)."""
    prompt_ids = tokenizer(example_prompt)["input_ids"]
    ids = {}
    for label, text in label_texts.items():
        full = tokenizer(example_prompt + text)["input_ids"]
        added = full[len(prompt_ids):]
        if full[:len(prompt_ids)] != prompt_ids or len(added) != 1:
            raise ValueError(f"label {text!r} is not a single token after the prompt: {added}")
        ids[int(label)] = added[0]
    return ids


def load_base_model(cfg: ModelConfig, compute_dtype):
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kwargs = {dtype_key: compute_dtype}
    if cfg.load_in_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype)
        kwargs["device_map"] = {"": 0}
    return AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)


def attach_lora(base_model, cfg: ModelConfig):
    from peft import LoraConfig, get_peft_model
    try:    # this project never uses torchao; stop peft from rejecting Colab's old preinstalled copy
        import peft.tuners.lora.torchao as _peft_torchao
        _peft_torchao.is_torchao_available = lambda: False
    except (ImportError, AttributeError):
        pass
    if cfg.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base_model.enable_input_require_grads()
    base_model.config.use_cache = False
    present = {n.split(".")[-1] for n, _ in base_model.named_modules()}
    targets = [t for t in cfg.target_modules if t in present]
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      bias="none", task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(base_model, lora)
    for p in model.parameters():              # adapters train in float32 for stable fp16 mixed precision
        if p.requires_grad:
            p.data = p.data.float()
    return model


def binary_logits(model, input_ids, attention_mask, label_ids: dict) -> torch.Tensor:
    """(batch, 2) logits for [label 0, label 1] at the final position."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    body = getattr(base, base.base_model_prefix)
    hidden = body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, -1, :]
    weight = base.get_output_embeddings().weight[[label_ids[0], label_ids[1]]]
    return hidden.float() @ weight.float().T


def count_parameters(model) -> dict:
    """4-bit weights are stored packed (2 per stored element), so count them twice."""
    total = trainable = 0
    for p in model.parameters():
        n = p.numel() * (2 if p.__class__.__name__ == "Params4bit" else 1)
        total += n
        trainable += p.numel() if p.requires_grad else 0
    return {"total_parameters": int(total), "trainable_parameters": int(trainable),
            "trainable_percent": round(100 * trainable / max(total, 1), 4)}


logger = logging.getLogger("TRAINING")


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_tokens: int = 256
    eval_every_steps: int = 100           # optimizer steps (keep a multiple of checkpoint_every_steps)
    checkpoint_every_steps: int = 100     # optimizer steps
    eval_batch_size: int = 32
    selection_metric: str = "pr_auc"      # weighted validation metric used to keep the best checkpoint
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Encoded:
    """Tokenised prompts kept as Python lists; padded per batch (dynamic padding)."""

    def __init__(self, tokenizer, prompts, labels, max_tokens: int):
        self.ids = tokenizer(list(prompts), add_special_tokens=True)["input_ids"]
        too_long = sum(len(x) > max_tokens for x in self.ids)
        if too_long:
            raise ValueError(f"{too_long} prompts exceed max_tokens={max_tokens}; refusing to truncate the answer slot")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.ids)

    def batch(self, idx, device):
        seqs = [self.ids[i] for i in idx]
        width = max(map(len, seqs))
        input_ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, s in enumerate(seqs):                      # left padding
            input_ids[r, width - len(s):] = torch.tensor(s)
            mask[r, width - len(s):] = 1
        return input_ids.to(device), mask.to(device), torch.tensor(self.labels[idx]).to(device)


@torch.no_grad()
def predict(model, data: Encoded, label_ids, device, dtype, batch_size: int = 32) -> np.ndarray:
    was_training = model.training
    model.eval()
    order = np.argsort([len(x) for x in data.ids])        # similar lengths together -> less padding
    probs = np.empty(len(data), dtype=np.float64)
    use_amp = device == "cuda"
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        ids, mask, _ = data.batch(idx, device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            logits = binary_logits(model, ids, mask, label_ids)
        probs[idx] = torch.softmax(logits, dim=-1)[:, 1].double().cpu().numpy()
    if was_training:
        model.train()
    return probs


def _adapter_state(model):
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}


def _load_adapter(model, state):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state)


def _atomic_save(target: Path, writer):
    """Overwrite checkpoint files in place so Google Drive does not fill its trash with old copies.

    trainer_state.json is removed first and written last by the writer: if a save is interrupted,
    the checkpoint is ignored on resume instead of being half old, half new.
    """
    target.mkdir(parents=True, exist_ok=True)
    state_file = target / "trainer_state.json"
    if state_file.exists():
        state_file.unlink()
    writer(target)


class Trainer:
    def __init__(self, model, tokenizer, label_ids: dict, cfg: TrainConfig, out_dir: Path, device: str, dtype):
        self.model, self.tok, self.label_ids, self.cfg = model, tokenizer, label_ids, cfg
        self.out_dir, self.device, self.dtype = Path(out_dir), device, dtype
        self.use_scaler = device == "cuda" and dtype == torch.float16

    def fit(self, train_prompts, train_labels, val_prompts, val_labels, val_weights=None,
            stop_after_steps: int | None = None) -> dict:
        cfg = self.cfg
        set_seed(cfg.seed)
        train = Encoded(self.tok, train_prompts, train_labels, cfg.max_tokens)
        val = Encoded(self.tok, val_prompts, val_labels, cfg.max_tokens)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        batches_per_epoch = math.ceil(len(train) / cfg.batch_size)
        total_micro = cfg.epochs * batches_per_epoch
        total_steps = math.ceil(total_micro / cfg.gradient_accumulation_steps)
        warmup = max(1, int(cfg.warmup_ratio * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total_steps - s) / max(total_steps - warmup, 1))
            if s >= warmup else (s + 1) / warmup)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        state = {"micro_step": 0, "opt_step": 0, "best_metric": -1.0, "best_step": None,
                 "train_seconds": 0.0, "samples_seen": 0, "resumed_from": None, "history": []}
        last = self.out_dir / "last_checkpoint"
        if (last / "trainer_state.json").exists():
            state = json.loads((last / "trainer_state.json").read_text())
            _load_adapter(self.model, torch.load(last / "adapter.pt", weights_only=False))
            extra = torch.load(last / "optimizer.pt", weights_only=False)
            opt.load_state_dict(extra["optimizer"]); sched.load_state_dict(extra["scheduler"])
            scaler.load_state_dict(extra["scaler"]); torch.set_rng_state(extra["torch_rng"])
            if torch.cuda.is_available() and extra.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(extra["cuda_rng"])
            state["resumed_from"] = state["opt_step"]
            logger.info("Resumed from optimizer step %d", state["opt_step"])

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        t_segment = time.time()
        accum_loss = 0.0
        for epoch in range(cfg.epochs):
            perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg.seed + epoch)).numpy()
            for b in range(batches_per_epoch):
                global_micro = epoch * batches_per_epoch + b
                if global_micro < state["micro_step"]:
                    continue                                    # already done before the interruption
                idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                ids, mask, y = train.batch(idx, self.device)
                with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.device == "cuda"):
                    logits = binary_logits(self.model, ids, mask, self.label_ids)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss / cfg.gradient_accumulation_steps).backward()
                accum_loss += loss.item() / cfg.gradient_accumulation_steps
                running_correct += int((logits.argmax(-1) == y).sum()); running_n += len(idx)
                state["micro_step"] += 1
                state["samples_seen"] += len(idx)

                if state["micro_step"] % cfg.gradient_accumulation_steps and state["micro_step"] != total_micro:
                    continue
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                state["opt_step"] += 1
                running_loss += accum_loss; accum_loss = 0.0
                step = state["opt_step"]

                is_eval = step % cfg.eval_every_steps == 0 or step == total_steps
                is_ckpt = step % cfg.checkpoint_every_steps == 0 or step == total_steps
                if is_eval or is_ckpt:
                    state["train_seconds"] += time.time() - t_segment
                    entry = {"opt_step": step, "epoch": round(state["micro_step"] / batches_per_epoch, 3),
                             "lr": sched.get_last_lr()[0], "train_seconds": round(state["train_seconds"], 1),
                             "samples_per_second": round(state["samples_seen"] / max(state["train_seconds"], 1e-9), 3)}
                    if running_n:
                        entry["train_loss"] = running_loss / max(1, step - (state["history"][-1]["opt_step"]
                                                                            if state["history"] else 0))
                        entry["train_accuracy"] = running_correct / running_n
                    if is_eval:
                        t_eval = time.time()
                        probs = predict(self.model, val, self.label_ids, self.device, self.dtype, cfg.eval_batch_size)
                        vm = classification_metrics(val.labels, probs, val_weights, 0.5)
                        entry.update({"val_loss": vm["log_loss"], "val_pr_auc": vm["pr_auc"], "val_roc_auc": vm["roc_auc"],
                                      "val_f1_at_0_5": vm["f1"], "val_accuracy": vm["accuracy"],
                                      "eval_seconds": round(time.time() - t_eval, 1)})
                        metric = vm.get(cfg.selection_metric) or -1.0
                        if metric > state["best_metric"]:
                            state["best_metric"], state["best_step"] = metric, step
                            _atomic_save(self.out_dir / "best_model",
                                         lambda d: torch.save(_adapter_state(self.model), d / "adapter.pt"))
                        logger.info("step %d/%d | loss %.4f | val PR-AUC %.4f | val loss %.4f", step, total_steps,
                                    entry.get("train_loss", float("nan")), vm["pr_auc"] or float("nan"), vm["log_loss"])
                    state["history"].append(entry)
                    running_loss, running_correct, running_n = 0.0, 0, 0
                    if is_ckpt:
                        extra = {"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                                 "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

                        def write(d, extra=extra):
                            torch.save(_adapter_state(self.model), d / "adapter.pt")
                            torch.save(extra, d / "optimizer.pt")
                            (d / "trainer_state.json").write_text(json.dumps(state, indent=1))
                        _atomic_save(last, write)
                    t_segment = time.time()
                if stop_after_steps is not None and step >= stop_after_steps and step < total_steps:
                    raise KeyboardInterrupt(f"simulated interruption at step {step}")

        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        return {"total_optimizer_steps": total_steps, "best_step": state["best_step"],
                "best_val_metric": state["best_metric"], "selection_metric": cfg.selection_metric,
                "train_seconds": round(state["train_seconds"], 1), "samples_seen": state["samples_seen"],
                "samples_per_second": round(state["samples_seen"] / max(state["train_seconds"], 1e-9), 3),
                "peak_gpu_memory_gb": None if peak is None else round(peak, 2),
                "resumed_from_step": state["resumed_from"], "history": state["history"]}
