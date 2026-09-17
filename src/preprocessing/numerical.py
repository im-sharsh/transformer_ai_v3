"""Numeric transformations fitted on training rows only: robust scaling, log transform, quantile buckets."""
from __future__ import annotations

import numpy as np
import pandas as pd


class NumericTransformer:
    def __init__(self, robust_scale: bool = True, log_skew_threshold: float = 2.0, n_bins: int = 10):
        self.robust_scale = robust_scale
        self.log_skew_threshold = log_skew_threshold
        self.n_bins = n_bins
        self.params: dict = {}

    def fit(self, train: pd.DataFrame, columns: list) -> "NumericTransformer":
        for c in columns:
            x = pd.to_numeric(train[c], errors="coerce").astype("float64").dropna()
            q25, q50, q75 = x.quantile([0.25, 0.5, 0.75])
            iqr = float(q75 - q25) or float(x.std()) or 1.0
            skew = float(x.skew())
            edges = np.unique(x.quantile(np.linspace(0, 1, self.n_bins + 1)[1:-1]).values).tolist()
            self.params[c] = {"median": float(q50), "iqr": iqr, "skewness": skew, "min": float(x.min()),
                              "log": bool(skew > self.log_skew_threshold and x.min() >= 0),
                              "bin_edges": edges}
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df, stats = df.copy(), {}
        for c, p in self.params.items():
            x = pd.to_numeric(df[c], errors="coerce").astype("float64")
            if self.robust_scale:
                df[f"{c}__robust"] = (x - p["median"]) / p["iqr"]
            if p["log"]:
                df[f"{c}__log"] = np.log1p(x.clip(lower=0))
            df[f"{c}__bin"] = pd.Series(np.searchsorted(p["bin_edges"], x, side="right"), index=df.index) \
                .astype("Int64").where(x.notna())
            stats[c] = {"outside_train_range": int(((x < p["min"])).sum())}
        return df, stats
