"""Fixed experiment samples with inverse-probability weights.

The same sample (same row IDs) is used by every condition and every training seed,
so differences between conditions cannot come from which rows were drawn.
Weights let metrics on an enriched sample estimate performance at the real fraud rate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class SampleConfig:
    sizes: dict = field(default_factory=lambda: {"train": 6000, "validation": 800, "test": 6000})
    positive_share: dict = field(default_factory=lambda: {"train": 0.10, "validation": 0.10})
    include_all_positives: dict = field(default_factory=lambda: {"test": True})
    max_positive_share: float = 0.5
    seed: int = 2024

    def to_dict(self):
        return asdict(self)


def draw_sample(frame: pd.DataFrame, target: str, cfg: SampleConfig) -> pd.DataFrame:
    """frame needs columns: _row_id, split, <target>. Returns rows with a _weight column."""
    parts = []
    for i, (split, size) in enumerate(cfg.sizes.items()):
        pool = frame[frame["split"] == split]
        y = pd.to_numeric(pool[target], errors="coerce")
        pos, neg = pool[y == 1], pool[y == 0]
        if cfg.include_all_positives.get(split):
            n_pos = min(len(pos), int(size * cfg.max_positive_share))
        elif split in cfg.positive_share:
            n_pos = min(len(pos), round(size * cfg.positive_share[split]))
        else:
            n_pos = min(len(pos), round(size * len(pos) / max(len(pool), 1)))
        n_neg = min(len(neg), size - n_pos)
        sp = pos.sample(n=n_pos, random_state=cfg.seed + i) if n_pos else pos.iloc[0:0]
        sn = neg.sample(n=n_neg, random_state=cfg.seed + 100 + i) if n_neg else neg.iloc[0:0]
        sp = sp.assign(_weight=len(pos) / max(n_pos, 1))
        sn = sn.assign(_weight=len(neg) / max(n_neg, 1))
        parts.append(pd.concat([sp, sn]).sample(frac=1, random_state=cfg.seed + 200 + i))
    return pd.concat(parts, ignore_index=True)


def sample_report(sample: pd.DataFrame, target: str) -> dict:
    out = {}
    for split, g in sample.groupby("split", sort=False):
        y = pd.to_numeric(g[target])
        out[split] = {"rows": len(g), "positives": int(y.sum()), "sample_positive_share": round(float(y.mean()), 4),
                      "weighted_positive_rate": round(float((g["_weight"] * y).sum() / g["_weight"].sum()), 6),
                      "represents_rows": int(round(g["_weight"].sum()))}
    return out


def redraw_split(sample: pd.DataFrame, frame: pd.DataFrame, target: str, split: str, size: int,
                 positive_share: float, seed: int) -> pd.DataFrame:
    """Replace one split of an existing sample with a new draw; every other split is kept row for row."""
    cfg = SampleConfig(sizes={split: size}, positive_share={split: positive_share}, include_all_positives={}, seed=seed)
    new = draw_sample(frame, target, cfg)
    kept = sample[sample["split"] != split]
    cols = [c for c in kept.columns if c in new.columns]
    return pd.concat([kept, new[cols]], ignore_index=True)
