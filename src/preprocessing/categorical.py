"""Rare-category grouping fitted on training rows only."""
from __future__ import annotations

import pandas as pd


class RareCategoryGrouper:
    def __init__(self, min_frequency: float = 0.001, rare_label: str = "__RARE__", unseen_label: str = "__UNSEEN__"):
        self.min_frequency = min_frequency
        self.rare_label = rare_label
        self.unseen_label = unseen_label
        self.params: dict = {}

    def fit(self, train: pd.DataFrame, columns: list) -> "RareCategoryGrouper":
        for c in columns:
            freq = train[c].astype("string").value_counts(normalize=True, dropna=True)
            self.params[c] = {"kept": sorted(freq[freq >= self.min_frequency].index.tolist()),
                              "seen": sorted(freq.index.tolist()),
                              "train_categories": int(len(freq))}
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df, stats = df.copy(), {}
        for c, p in self.params.items():
            s = df[c].astype("string")
            kept, seen = s.isin(p["kept"]), s.isin(p["seen"])
            out = s.astype("object")
            out[s.notna() & ~kept & seen] = self.rare_label
            out[s.notna() & ~seen] = self.unseen_label
            df[c] = out
            stats[c] = {"rare": int((s.notna() & ~kept & seen).sum()), "unseen": int((s.notna() & ~seen).sum())}
        return df, stats
