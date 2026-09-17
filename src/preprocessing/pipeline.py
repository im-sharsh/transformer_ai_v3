"""Train-fitted preprocessing: fit on training rows, apply the saved parameters to every split."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.preprocessing.categorical import RareCategoryGrouper
from src.preprocessing.datetime_features import add_time_features
from src.preprocessing.missing_values import MissingValueHandler
from src.preprocessing.numerical import NumericTransformer

logger = logging.getLogger("PREPROCESSING")
PIPELINE_VERSION = "0.2.0"


@dataclass
class PreprocessingConfig:
    time_features: bool = True
    missing_values: bool = True
    rare_categories: bool = True
    numeric_transforms: bool = True
    rare_min_frequency: float = 0.001
    log_skew_threshold: float = 2.0
    n_bins: int = 10

    def to_dict(self):
        return asdict(self)


@dataclass
class ColumnRoles:
    numeric: list
    categorical: list
    event_time: str | None
    birth_dates: list = field(default_factory=list)
    target: str | None = None

    @classmethod
    def from_schema(cls, schema) -> "ColumnRoles":
        roles = schema.by_role()
        datetimes = roles.get("datetime", [])
        return cls(numeric=roles.get("numeric", []),
                   categorical=roles.get("categorical", []) + roles.get("categorical_code", []),
                   event_time=next((c for c in datetimes if schema.columns[c].pii_type != "birth_date"), None),
                   birth_dates=[c for c in datetimes if schema.columns[c].pii_type == "birth_date"],
                   target=schema.target)


class FittedPreprocessor:
    def __init__(self, config: PreprocessingConfig | None = None):
        self.cfg = config or PreprocessingConfig()
        self.roles: ColumnRoles | None = None
        self.time_columns: list = []
        self.missing = MissingValueHandler()
        self.categorical = RareCategoryGrouper(self.cfg.rare_min_frequency)
        self.numeric = NumericTransformer(log_skew_threshold=self.cfg.log_skew_threshold, n_bins=self.cfg.n_bins)
        self.fit_info: dict = {}

    def fit(self, train: pd.DataFrame, roles: ColumnRoles) -> "FittedPreprocessor":
        t0 = time.time()
        self.roles = roles
        if self.cfg.time_features and roles.event_time:
            train, self.time_columns = add_time_features(train, roles.event_time, roles.birth_dates)
        if self.cfg.missing_values:
            self.missing.fit(train, roles.numeric + self.time_columns, roles.categorical)
        if self.cfg.rare_categories:
            self.categorical.fit(train, roles.categorical)
        if self.cfg.numeric_transforms:
            self.numeric.fit(train, roles.numeric)
        self.fit_info = {"fitted_on_rows": len(train), "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "seconds": round(time.time() - t0, 1)}
        logger.info("Fitted on %d training rows", len(train))
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        stats = {}
        if self.cfg.time_features and self.roles.event_time:
            df, _ = add_time_features(df, self.roles.event_time, self.roles.birth_dates)
        if self.cfg.missing_values:
            df, stats["missing_filled"] = self.missing.transform(df)
        if self.cfg.rare_categories:
            df, stats["categories"] = self.categorical.transform(df)
        if self.cfg.numeric_transforms:
            df, stats["numeric"] = self.numeric.transform(df)
        return df, stats

    def to_dict(self) -> dict:
        return {"pipeline_version": PIPELINE_VERSION, "config": self.cfg.to_dict(), "roles": asdict(self.roles),
                "time_columns": self.time_columns, "fit_info": self.fit_info,
                "missing_values": self.missing.params, "categorical": self.categorical.params,
                "numeric": self.numeric.params}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FittedPreprocessor":
        d = json.loads(Path(path).read_text())
        obj = cls(PreprocessingConfig(**d["config"]))
        obj.roles, obj.time_columns, obj.fit_info = ColumnRoles(**d["roles"]), d["time_columns"], d["fit_info"]
        obj.missing.params, obj.categorical.params, obj.numeric.params = d["missing_values"], d["categorical"], d["numeric"]
        return obj
