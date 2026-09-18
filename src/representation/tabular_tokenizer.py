"""Turn a row of mixed tabular features into a fixed-length sequence for the model.

Each feature is one sequence position. Position 0 is [CLS]. Categorical values map to per-feature value
tokens (rare -> unknown, missing -> missing). The same tokenizer settings are used for every data level, so
the representation stays constant while the data preparation changes.

numeric_mode="quantile_bin" (the original, default): numeric values also map to per-feature quantile-bin
tokens, exactly like a categorical feature. transform() returns one array of integer token ids.

numeric_mode="continuous" (audit R2/R3 experiment, opt-in, added after measuring finding 1: quantile-binning
throws away the numeric magnitude, keeping only its rank -- see levels.py's numeric-redundancy comment for the
proof on the *duplicated* bin case). Numeric values are standardized (train-fit median/IQR) and kept as
floats; transform() still returns token ids (a per-feature "missing" marker, a coarse quantile-bin id if
coarse_bins > 0 -- the R3 variant -- or PAD otherwise), and the new transform_numeric() returns the parallel
(values, mask) arrays the model actually reads the magnitude from. This is strictly opt-in: the default mode
and its output shape/dtype are unchanged from before this feature existed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAD, CLS = 0, 1


class TabularTokenizer:
    def __init__(self, numeric_bins: int = 16, min_category_count: int = 5, numeric_mode: str = "quantile_bin",
                coarse_bins: int = 0):
        if numeric_mode not in ("quantile_bin", "continuous"):
            raise ValueError(f"numeric_mode must be 'quantile_bin' or 'continuous', got {numeric_mode!r}")
        self.numeric_bins, self.min_count = numeric_bins, min_category_count
        self.numeric_mode, self.coarse_bins = numeric_mode, int(coarse_bins)
        self.numeric, self.categorical = [], []
        self.edges: dict = {}
        self.vocab: dict = {}          # feature -> {value_or_bin: token}
        self.special: dict = {}        # feature -> {"missing": id[, "unknown": id]}
        self.numeric_stats: dict = {}  # continuous mode only: feature -> {"median":, "scale":}
        self.vocab_size = 2

    def _new(self) -> int:
        self.vocab_size += 1
        return self.vocab_size - 1

    def fit(self, train: pd.DataFrame, numeric: list, categorical: list) -> "TabularTokenizer":
        self.numeric, self.categorical = list(numeric), list(categorical)
        for c in self.numeric:
            x = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            self.special[c] = {"missing": self._new()}
            if self.numeric_mode == "quantile_bin":
                edges = np.unique(np.quantile(x, np.linspace(0, 1, self.numeric_bins + 1)[1:-1])) if len(x) else np.array([])
                self.edges[c] = edges
                self.vocab[c] = {b: self._new() for b in range(len(edges) + 1)}
            else:
                median = float(x.median()) if len(x) else 0.0
                if len(x):
                    q1, q3 = x.quantile(0.25), x.quantile(0.75)
                    iqr = float(q3 - q1)
                else:
                    iqr = 0.0
                scale = iqr if iqr > 1e-9 else (float(x.std()) if len(x) and x.std() > 1e-9 else 1.0)
                self.numeric_stats[c] = {"median": median, "scale": scale}
                if self.coarse_bins > 0:
                    edges = np.unique(np.quantile(x, np.linspace(0, 1, self.coarse_bins + 1)[1:-1])) if len(x) else np.array([])
                    self.edges[c] = edges
                    self.vocab[c] = {b: self._new() for b in range(len(edges) + 1)}
        for c in self.categorical:
            self.special[c] = {"missing": self._new(), "unknown": self._new()}
            counts = train[c].astype("string").value_counts()
            self.vocab[c] = {v: self._new() for v in counts[counts >= self.min_count].index}
        return self

    @property
    def n_positions(self) -> int:
        return 1 + len(self.numeric) + len(self.categorical)

    @property
    def numeric_positions(self) -> list:
        """Sequence positions (0-indexed, CLS at 0) that carry a numeric value in continuous mode."""
        return list(range(1, 1 + len(self.numeric)))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), self.n_positions), dtype=np.int64)
        out[:, 0] = CLS
        for j, c in enumerate(self.numeric, start=1):
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
            bad = ~np.isfinite(x)
            binned = self.numeric_mode == "quantile_bin" or self.coarse_bins > 0
            if binned:
                bins = np.searchsorted(self.edges[c], np.where(bad, 0, x), side="right")
                ids = np.vectorize(self.vocab[c].get)(bins) if len(x) else np.array([], dtype=np.int64)
                out[:, j] = np.where(bad, self.special[c]["missing"], ids)
            else:
                out[:, j] = np.where(bad, self.special[c]["missing"], PAD)
        for j, c in enumerate(self.categorical, start=1 + len(self.numeric)):
            s = df[c].astype("string")
            mapped = s.map(self.vocab[c]).astype("float64")
            ids = mapped.fillna(self.special[c]["unknown"]).to_numpy()
            out[:, j] = np.where(s.isna().to_numpy(), self.special[c]["missing"], ids).astype(np.int64)
        return out

    def transform_numeric(self, df: pd.DataFrame) -> tuple:
        """Continuous mode only. Returns (values, mask), each shape (len(df), n_positions), float32.
        `values` is the standardized numeric value at numeric positions (0 elsewhere and where missing);
        `mask` is 1.0 at a present numeric position, 0.0 elsewhere (including CLS/categorical positions
        and missing numeric values) -- the model multiplies its numeric contribution by this mask."""
        if self.numeric_mode != "continuous":
            raise RuntimeError("transform_numeric() is only meaningful in numeric_mode='continuous'")
        n = len(df)
        values = np.zeros((n, self.n_positions), dtype=np.float32)
        mask = np.zeros((n, self.n_positions), dtype=np.float32)
        for j, c in enumerate(self.numeric, start=1):
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
            good = np.isfinite(x)
            stats = self.numeric_stats[c]
            standardized = (x - stats["median"]) / stats["scale"]
            values[:, j] = np.where(good, standardized, 0.0).astype(np.float32)
            mask[:, j] = good.astype(np.float32)
        return values, mask

    def to_dict(self) -> dict:
        return {"numeric_bins": self.numeric_bins, "min_category_count": self.min_count,
                "numeric_mode": self.numeric_mode, "coarse_bins": self.coarse_bins,
                "numeric": self.numeric, "categorical": self.categorical,
                "edges": {c: e.tolist() for c, e in self.edges.items()},
                "vocab": {c: {str(k): v for k, v in m.items()} for c, m in self.vocab.items()},
                "special": self.special, "numeric_stats": self.numeric_stats, "vocab_size": self.vocab_size}

    @classmethod
    def from_dict(cls, d: dict) -> "TabularTokenizer":
        tok = cls(d["numeric_bins"], d["min_category_count"], d.get("numeric_mode", "quantile_bin"),
                  d.get("coarse_bins", 0))
        tok.numeric, tok.categorical, tok.special, tok.vocab_size = d["numeric"], d["categorical"], d["special"], d["vocab_size"]
        tok.numeric_stats = d.get("numeric_stats", {})
        tok.edges = {c: np.array(e) for c, e in d["edges"].items()}
        tok.vocab = {c: ({int(k): v for k, v in m.items()} if c in tok.numeric else dict(m)) for c, m in d["vocab"].items()}
        return tok
