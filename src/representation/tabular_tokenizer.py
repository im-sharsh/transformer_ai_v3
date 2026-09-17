"""Turn a row of mixed tabular features into a fixed-length sequence of token ids.

Each feature becomes one token: categorical values map to per-feature value tokens (rare -> unknown), numeric
values map to per-feature quantile-bin tokens fitted on training rows. Missing values get a per-feature
missing token. Position 0 is a [CLS] token. The same tokenizer settings are used for every data level,
so the representation stays constant while the data preparation changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAD, CLS = 0, 1


class TabularTokenizer:
    def __init__(self, numeric_bins: int = 16, min_category_count: int = 5):
        self.numeric_bins, self.min_count = numeric_bins, min_category_count
        self.numeric, self.categorical = [], []
        self.edges: dict = {}
        self.vocab: dict = {}          # feature -> {value: token}
        self.special: dict = {}        # feature -> {"missing": id, "unknown": id}
        self.vocab_size = 2

    def _new(self) -> int:
        self.vocab_size += 1
        return self.vocab_size - 1

    def fit(self, train: pd.DataFrame, numeric: list, categorical: list) -> "TabularTokenizer":
        self.numeric, self.categorical = list(numeric), list(categorical)
        for c in self.numeric + self.categorical:
            self.special[c] = {"missing": self._new(), "unknown": self._new()}
        for c in self.numeric:
            x = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            edges = np.unique(np.quantile(x, np.linspace(0, 1, self.numeric_bins + 1)[1:-1])) if len(x) else np.array([])
            self.edges[c] = edges
            self.vocab[c] = {b: self._new() for b in range(len(edges) + 1)}
        for c in self.categorical:
            counts = train[c].astype("string").value_counts()
            self.vocab[c] = {v: self._new() for v in counts[counts >= self.min_count].index}
        return self

    @property
    def n_positions(self) -> int:
        return 1 + len(self.numeric) + len(self.categorical)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), self.n_positions), dtype=np.int64)
        out[:, 0] = CLS
        for j, c in enumerate(self.numeric, start=1):
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
            bad = ~np.isfinite(x)
            bins = np.searchsorted(self.edges[c], np.where(bad, 0, x), side="right")
            ids = np.vectorize(self.vocab[c].get)(bins) if len(x) else np.array([], dtype=np.int64)
            out[:, j] = np.where(bad, self.special[c]["missing"], ids)
        for j, c in enumerate(self.categorical, start=1 + len(self.numeric)):
            s = df[c].astype("string")
            mapped = s.map(self.vocab[c]).astype("float64")
            ids = mapped.fillna(self.special[c]["unknown"]).to_numpy()
            out[:, j] = np.where(s.isna().to_numpy(), self.special[c]["missing"], ids).astype(np.int64)
        return out

    def to_dict(self) -> dict:
        return {"numeric_bins": self.numeric_bins, "min_category_count": self.min_count, "numeric": self.numeric,
                "categorical": self.categorical, "edges": {c: e.tolist() for c, e in self.edges.items()},
                "vocab": {c: {str(k): v for k, v in m.items()} for c, m in self.vocab.items()},
                "special": self.special, "vocab_size": self.vocab_size}

    @classmethod
    def from_dict(cls, d: dict) -> "TabularTokenizer":
        tok = cls(d["numeric_bins"], d["min_category_count"])
        tok.numeric, tok.categorical, tok.special, tok.vocab_size = d["numeric"], d["categorical"], d["special"], d["vocab_size"]
        tok.edges = {c: np.array(e) for c, e in d["edges"].items()}
        tok.vocab = {c: ({int(k): v for k, v in m.items()} if c in tok.numeric else dict(m)) for c, m in d["vocab"].items()}
        return tok
