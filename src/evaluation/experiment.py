"""Controlled experiments: run the same model, seed and settings on E0, E1 and E2, sharing the same split and
sample (from a single `DataPreparer`), so any difference in results comes from the data preparation, not the
training setup. Answers the research question directly: does preprocessing change model performance?
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.models.sanity_transformer import SanityTransformerAdapter
from src.preprocessing.levels import DataPreparer
from src.utils.audit import next_version

LEVEL_ORDER = ["E0", "E1", "E2"]


@dataclass
class ExperimentResult:
    level: str
    level_name: str
    rows: dict                    # split -> row count
    features: int
    model: str
    device: str
    seed: int
    settings: dict
    train_seconds: float
    inference_seconds: float
    epochs_run: int
    threshold: float
    metrics: dict                 # validation / test / test_unweighted / test_known_entities (if present)
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def run_experiment(preparer: DataPreparer, level: str, config: dict, settings: dict | None = None,
                   device: str | None = None) -> ExperimentResult:
    """Builds (or reuses, if already built) `level` from `preparer`, trains and evaluates one model on it."""
    prepared = preparer.build(level)
    adapter = SanityTransformerAdapter(config, device=device)
    t0 = time.time()
    summary = adapter.train(prepared, settings or {})
    t1 = time.time()
    results, _, _ = adapter.evaluate(prepared)
    t2 = time.time()
    return ExperimentResult(level=level, level_name=prepared.info["name"],
                            rows={s: len(f) for s, f in prepared.frames.items()}, features=len(prepared.features),
                            model=adapter.label, device=adapter.describe()["device"],
                            seed=int(summary["settings"]["seed"]), settings=summary["settings"],
                            train_seconds=round(t1 - t0, 2), inference_seconds=round(t2 - t1, 2),
                            epochs_run=summary["epochs_run"], threshold=results["threshold_from_validation"],
                            metrics=results, created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


def run_comparison(preparer: DataPreparer, levels: list, config: dict, settings: dict | None = None,
                   device: str | None = None, progress=None) -> list[ExperimentResult]:
    out = []
    for i, level in enumerate(sorted(levels, key=lambda l: LEVEL_ORDER.index(l))):
        r = run_experiment(preparer, level, config, settings, device)
        out.append(r)
        if progress:
            progress(i + 1, len(levels), r)
    return out


def run_comparison_multiseed(preparer: DataPreparer, levels: list, config: dict, seeds: list, settings: dict | None = None,
                             device: str | None = None, progress=None) -> dict:
    """Repeats run_comparison once per seed (same split/sample/settings throughout — only the seed changes),
    so E0/E1/E2 can be reported as mean ± std rather than trusted from a single run (see the project brief's
    own §16 requirement, not previously implemented: run_comparison always used exactly one seed).
    Returns {seed: [ExperimentResult, ...]}. Levels already built on `preparer` are reused across seeds
    (DataPreparer caches by level), so only training + evaluation repeat per seed, not data preparation."""
    out = {}
    total = len(seeds) * len(levels)
    done = 0
    for seed in seeds:
        per_seed_settings = {**(settings or {}), "seed": int(seed)}

        def _progress(i, n, r, seed=seed):
            nonlocal done
            done += 1
            if progress:
                progress(done, total, seed, r)
        out[seed] = run_comparison(preparer, levels, config, per_seed_settings, device, _progress if progress else None)
    return out


def aggregate_seeds(results_by_seed: dict) -> pd.DataFrame:
    """Mean +/- std of the test-set metrics across seeds, one row per level, in E0/E1/E2 order.
    Never fabricates a std for a single seed (reports it as 0.0 explicitly rather than NaN-hiding it)."""
    metric_keys = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    rows = []
    for seed, results in results_by_seed.items():
        for r in results:
            m = r.metrics["test"]
            rows.append({"seed": seed, "Level": r.level, **{k: m[k] for k in metric_keys}})
    long = pd.DataFrame(rows)
    n_seeds = long.groupby("Level")["seed"].nunique()
    agg = long.groupby("Level")[metric_keys].agg(["mean", "std"])
    agg.columns = [f"{k} {stat}" for k, stat in agg.columns]
    for k in metric_keys:
        agg[f"{k} std"] = agg[f"{k} std"].fillna(0.0)
    agg.insert(0, "Seeds", n_seeds)
    return agg.reindex(sorted(agg.index, key=lambda l: LEVEL_ORDER.index(l)))


def comparison_table(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.metrics["test"]
        rows.append({"Level": r.level, "Name": r.level_name, "Features": r.features, "Train rows": r.rows["train"],
                    "Epochs": r.epochs_run, "Train time (s)": r.train_seconds, "Device": r.device,
                    "Threshold": round(r.threshold, 4), "Test PR-AUC": m["pr_auc"], "Test ROC-AUC": m["roc_auc"],
                    "Test Precision": m["precision"], "Test Recall": m["recall"], "Test F1": m["f1"],
                    "Test positives": m["positives"]})
    return pd.DataFrame(rows)


def save_experiment(results: list, project_root: str | Path, dataset_id: str, split_info: dict) -> Path:
    """Writes one never-overwritten JSON file per comparison run, under experiments/."""
    project_root = Path(project_root)
    out_dir = project_root / "experiments"
    version = next_version(out_dir, prefix="experiment_v")
    path = out_dir / f"{version}.json"
    manifest = {"experiment": version, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset_id": dataset_id,
               "split": split_info, "results": [r.to_dict() for r in results]}
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def list_experiments(project_root: str | Path) -> list[dict]:
    """Every saved experiment manifest, most recent first. Returns [] if none exist yet — never fabricated."""
    out_dir = Path(project_root) / "experiments"
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("experiment_v*.json"), reverse=True)
    return [json.loads(p.read_text()) for p in files]
