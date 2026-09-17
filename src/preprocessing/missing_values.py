"""Missing-value handling fitted on training rows only."""
from __future__ import annotations

import numpy as np
import pandas as pd


class MissingValueHandler:
    def __init__(self, numeric_strategy: str = "median", categorical_fill: str = "UNKNOWN",
                 add_indicators: bool = True):
        self.numeric_strategy = numeric_strategy
        self.categorical_fill = categorical_fill
        self.add_indicators = add_indicators
        self.params: dict = {}

    def fit(self, train: pd.DataFrame, numeric: list, categorical: list) -> "MissingValueHandler":
        for c in numeric:
            x = pd.to_numeric(train[c], errors="coerce").astype("float64")
            fill = x.median() if self.numeric_strategy == "median" else x.mean()
            self.params[c] = {"type": "numeric", "strategy": self.numeric_strategy,
                              "fill": None if np.isnan(fill) else float(fill),
                              "train_missing_rate": float(x.isna().mean()),
                              "indicator": bool(self.add_indicators and x.isna().any())}
        for c in categorical:
            self.params[c] = {"type": "categorical", "strategy": "constant", "fill": self.categorical_fill,
                              "train_missing_rate": float(train[c].isna().mean()),
                              "indicator": bool(self.add_indicators and train[c].isna().any())}
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        df, stats = df.copy(), {}
        for c, p in self.params.items():
            missing = df[c].isna()
            if p["indicator"]:                    # the indicator set is decided on train, never on test
                df[f"{c}__was_missing"] = missing.astype("int8")
            if missing.any():
                if p["type"] == "numeric":
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64").fillna(p["fill"])
                else:
                    df[c] = df[c].astype("object").where(~missing, p["fill"])
                stats[c] = int(missing.sum())
        return df, stats
