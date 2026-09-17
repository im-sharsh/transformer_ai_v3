"""Data preparation levels.

E0  Raw                 minimal formatting: the same identifier / personal-data exclusion policy as every level
E1  Quality processed   type normalisation, validation and cleaning, categorical normalisation, rare-category
                        handling, numeric transformations, datetime processing (parsing, age from birth date)
E2  Feature engineered  E1 + hour / day of week / month / weekend (raw and cyclical) + point-in-time history
                        features (entity and merchant), skipped gracefully when no entity/time/amount is set

All levels share the same split and the same sampled rows. Everything statistical is fitted on training rows
only, and every history feature uses only transactions strictly earlier in time.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.behavioral_features import HistoryConfig, card_history, check_point_in_time, merchant_history
from src.ingestion.roles import DatasetRoles
from src.ingestion.schema_detector import SchemaReport
from src.preprocessing.cleaner import CleaningConfig, clean_dataset
from src.preprocessing.datetime_features import add_time_features
from src.preprocessing.pipeline import ColumnRoles, FittedPreprocessor, PreprocessingConfig
from src.preprocessing.sampling import SampleConfig, draw_sample, sample_report
from src.profiling.profiler import parse_datetime_column
from src.quality.quality_engine import assess_quality

LEVELS = {"E0": "Raw", "E1": "Quality processed", "E2": "Feature engineered"}
META = ["_row_id", "_split", "_target", "_weight", "_new_entity"]
PII_EXCLUDE = {"person_name", "address", "card_number", "account_number", "email", "phone", "government_id",
               "location", "birth_date"}
HINTS = {"amount": r"(^|_)(amt|amount|value|price|total|sum)($|_)", "category": r"(categ|mcc|(^|_)type($|_)|segment)",
         "merchant": r"(merchant|payee|store|vendor|shop)"}
HISTORY_FEATURES = ["card_seconds_since_prev", "card_count_1h", "card_count_24h", "card_count_7d", "card_count_30d",
                    "card_amount_sum_24h", "card_amount_mean_before", "card_amount_std_before",
                    "card_amount_ratio_to_mean", "card_amount_zscore", "card_first_time_category",
                    "card_category_share_before", "card_first_time_merchant", "merchant_count_24h",
                    "merchant_count_7d", "merchant_amount_mean_before", "merchant_amount_ratio_to_mean"]
TIME_FEATURES = ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"]


def _match(name: str, pattern: str) -> bool:
    return re.search(pattern, name.lower()) is not None


def _quasi_identifiers(df: pd.DataFrame, entity: str, candidates: list, min_stable_share: float = 0.95,
                       min_distinct_ratio: float = 0.2) -> dict:
    """Columns fixed per entity AND with many distinct values act as a stand-in for the entity itself
    (e.g. a job title or street address that happens to be unique per card)."""
    if not candidates:
        return {}
    n_entities = df[entity].nunique()
    nunique = df.groupby(entity)[candidates].nunique(dropna=True)
    found = {}
    for c in candidates:
        stable = float((nunique[c] <= 1).mean())
        distinct = int(df[c].nunique())
        if stable >= min_stable_share and distinct >= min_distinct_ratio * n_entities:
            found[c] = (f"quasi-identifier: fixed for {stable:.0%} of {entity} values and "
                        f"{distinct} distinct values for {n_entities} entities")
    return found


@dataclass
class Roles:
    target: str
    time: str | None
    entity: str | None
    amount: str | None
    category: str | None
    merchant: str | None
    birth: list
    excluded: dict                       # column -> reason; identical for every level

    def as_dict(self):
        return self.__dict__.copy()


def infer_roles(df: pd.DataFrame, schema: SchemaReport, dataset_roles: DatasetRoles, hints: dict | None = None,
                quasi_sample_rows: int = 200_000) -> Roles:
    """`schema` must be `schema_for_profiling()`'s output and `dataset_roles` the already-confirmed target /
    entity / datetime (chosen or detected on the Profiling page). Only amount / category / merchant and the
    identifier / personal-data / quasi-identifier exclusion policy are inferred here."""
    hints = {k: v for k, v in (hints or {}).items() if v}
    by_role = schema.by_role()
    cols = schema.columns
    target = dataset_roles.target
    if not target or target not in df:
        raise ValueError("No target column selected. Choose one on the Profiling page first.")
    if dataset_roles.task != "binary_classification":
        raise ValueError(f"E0 / E1 / E2 processing needs a binary target; '{target}' was detected as "
                         f"{dataset_roles.task or 'unknown'}. Change the task on the Profiling page.")
    time_col = dataset_roles.datetime
    entity = dataset_roles.entity
    numeric = [c for c in by_role.get("numeric", []) if c != target]
    categorical = [c for c in by_role.get("categorical", []) if not cols[c].pii_type]
    pick = lambda key, pool: hints.get(f"{key}_column") or next((c for c in pool if _match(c, HINTS[key])), None)
    roles = Roles(target=target, time=time_col, entity=entity, amount=pick("amount", numeric),
                  category=pick("category", categorical), merchant=pick("merchant", categorical),
                  birth=[c for c in by_role.get("datetime", []) if cols[c].pii_type == "birth_date"], excluded={})

    for role in ["record_id", "row_index", "entity_id", "constant"]:
        for c in by_role.get(role, []):
            if c != entity:
                roles.excluded[c] = f"{role.replace('_', ' ')}: carries no generalisable signal"
    for c, cs in cols.items():
        if cs.pii_type in PII_EXCLUDE and c not in roles.excluded:
            roles.excluded[c] = f"personal data ({cs.pii_type})"
    if entity and entity in df:
        candidates = [c for c in df.columns if c not in roles.excluded and c not in (target, time_col, entity)
                      and cols.get(c) is not None and cols[c].role not in ("datetime", "datetime_epoch", "free_text")]
        frame = df.sample(min(len(df), quasi_sample_rows), random_state=0) if len(df) > quasi_sample_rows else df
        for c, reason in _quasi_identifiers(frame, entity, candidates).items():
            if c not in (roles.amount, roles.category, roles.merchant):
                roles.excluded[c] = reason
    roles.excluded.pop(target, None)
    return roles


@dataclass
class PreparedLevel:
    level: str
    frames: dict                        # split -> DataFrame (META + features)
    numeric: list
    categorical: list
    info: dict = field(default_factory=dict)

    @property
    def features(self):
        return self.numeric + self.categorical


class DataPreparer:
    def __init__(self, df: pd.DataFrame, schema: SchemaReport, roles: Roles, config: dict, rows: int | str = 20000,
                 seed: int = 42):
        self.df = df.reset_index(drop=True)
        if "_row_id" not in self.df:
            self.df.insert(0, "_row_id", [f"r{i}" for i in range(len(self.df))])
        self.schema, self.roles, self.cfg, self.rows, self.seed = schema, roles, config, rows, seed
        self.sample_ids: pd.DataFrame | None = None
        self.split_info: dict = {}
        self._history: pd.DataFrame | None = None
        self.pit: dict | None = None
        self.cache: dict = {}

    # ------------------------------------------------------------------ target, split, sample
    def _binary_target(self) -> pd.Series:
        y = self.df[self.roles.target]
        num = pd.to_numeric(y, errors="coerce")
        if num.notna().mean() > 0.99 and set(num.dropna().unique()) <= {0, 1}:
            return num
        values = y.dropna().astype(str).value_counts()
        if len(values) != 2:
            raise ValueError(f"Target '{self.roles.target}' must have exactly two values; found {len(values)}")
        positive = values.index[-1]                                  # minority class is the positive class
        return (y.astype(str) == positive).astype(float).where(y.notna())

    def prepare_split(self) -> dict:
        t0 = time.time()
        cfg, df = self.cfg, self.df
        f_train, f_val, _ = cfg["split"]["fractions"]
        target = self._binary_target()
        split = pd.Series("train", index=df.index, dtype="object")
        if cfg["split"]["method"] == "temporal" and self.roles.time:
            t = parse_datetime_column(df[self.roles.time], "datetime")
            q1, q2 = t.quantile(f_train), t.quantile(f_train + f_val)
            split[t >= q1] = "validation"
            split[t >= q2] = "test"
            split[t.isna()] = "unassigned"
            boundaries = {"train": f"{t.min()} to before {q1}", "validation": f"{q1} to before {q2}",
                          "test": f"{q2} to {t.max()}", "method": "temporal (quantiles of event time)"}
        else:
            rng = np.random.default_rng(self.seed).random(len(df))
            split[rng >= f_train] = "validation"
            split[rng >= f_train + f_val] = "test"
            boundaries = {"method": "random (no datetime column selected)" if cfg["split"]["method"] == "temporal"
                          else "random (configured)"}
        split[target.isna()] = "unassigned"
        frame = pd.DataFrame({"_row_id": df["_row_id"], "split": split, "y": target})
        frame = frame[frame["split"] != "unassigned"]

        s = cfg["sampling"]
        if self.rows == "full" or int(self.rows) >= len(frame):
            sizes = frame["split"].value_counts().to_dict()
            scfg = SampleConfig(sizes={k: sizes.get(k, 0) for k in ["train", "validation", "test"]}, positive_share={},
                                include_all_positives={}, seed=self.seed)
        else:
            n = int(self.rows)
            sizes = {"train": int(n * f_train), "validation": int(n * f_val), "test": n - int(n * f_train) - int(n * f_val)}
            scfg = SampleConfig(sizes=sizes, positive_share={"train": s["train_positive_share"],
                                                             "validation": s["validation_positive_share"]},
                                include_all_positives={"test": s["keep_all_test_positives"]},
                                max_positive_share=s["max_test_positive_share"], seed=self.seed)
        sample = draw_sample(frame, "y", scfg)
        if self.roles.entity:
            train_entities = set(df.loc[split == "train", self.roles.entity])
            entity_of = df.set_index("_row_id")[self.roles.entity]
            sample["_new_entity"] = ~sample["_row_id"].map(entity_of).isin(train_entities)
        else:
            sample["_new_entity"] = False
        self.sample_ids = sample.rename(columns={"split": "_split", "y": "_target"})
        self.split_info = {"boundaries": boundaries, "rows_available": int(len(df)),
                           "rows_unassigned": int((split == "unassigned").sum()),
                           "mode_rows": self.rows, "sample": sample_report(sample.rename(columns={"y": "is_y"}), "is_y"),
                           "seconds": round(time.time() - t0, 2)}
        return self.split_info

    def _sample_rows(self) -> pd.DataFrame:
        if self.sample_ids is None:
            self.prepare_split()
        return self.sample_ids.merge(self.df, on="_row_id", how="left")

    # ------------------------------------------------------------------ levels
    def build(self, level: str) -> PreparedLevel:
        if level in self.cache:
            return self.cache[level]
        t0 = time.time()
        builder = {"E0": self._build_e0, "E1": self._build_e1, "E2": self._build_e2}[level]
        prepared = builder()
        prepared.info.update({"level": level, "name": LEVELS[level], "seconds": round(time.time() - t0, 2),
                              "features": len(prepared.features),
                              "rows": {s: int(len(f)) for s, f in prepared.frames.items()},
                              "positives": {s: int(f["_target"].sum()) for s, f in prepared.frames.items()},
                              "split": self.split_info})
        self.cache[level] = prepared
        return prepared

    def _finish(self, level, frame, features, steps, extra=None) -> PreparedLevel:
        features = [c for c in features if c in frame]
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
        categorical = [c for c in features if c not in numeric]
        frames = {s: frame[frame["_split"] == s][META + features].reset_index(drop=True) for s in ["train", "validation", "test"]}
        return PreparedLevel(level, frames, numeric, categorical, info={"steps": steps, **(extra or {})})

    def _build_e0(self) -> PreparedLevel:
        rows = self._sample_rows()
        r = self.roles
        features = [c for c in self.df.columns if c not in META and c not in r.excluded and c != r.target]
        steps = ["Minimal formatting: values kept exactly as loaded",
                 f"Excluded {len(r.excluded)} identifier / personal-data / quasi-identifier columns (same policy for every level)"]
        return self._finish("E0", rows, features, steps, {"excluded": r.excluded})

    def _e1_frame(self) -> tuple[pd.DataFrame, list, list, dict]:
        r, cfg = self.roles, self.cfg
        rows = self._sample_rows()
        content_cols = [c for c in self.df.columns if c != "_row_id"]
        content = rows[content_cols]
        quality = assess_quality(content, self.schema)
        protected = rows["_split"] != "train"
        cleaning = clean_dataset(rows[["_row_id"] + content_cols].assign(_split=rows["_split"]), self.schema, quality,
                                 protected=protected, config=CleaningConfig())
        cleaned = cleaning.df.merge(rows[["_row_id", "_target", "_weight", "_new_entity"]], on="_row_id", how="left")
        steps = [f"Quality score on sampled rows: {quality.scores['overall']:.1f}",
                 f"Cleaning: {cleaning.summary['rows_in']:,} rows in, {cleaning.summary['rows_quarantined']:,} quarantined, "
                 f"{cleaning.summary['duplicates_removed']:,} duplicates removed (validation/test rows are never removed)",
                 f"Dropped columns: {cleaning.summary['dropped_columns'] or 'none'}"]
        added = []
        if r.time and r.birth and r.time in cleaned:
            with_age, cols = add_time_features(cleaned[[r.time] + [b for b in r.birth if b in cleaned]], r.time,
                                               [b for b in r.birth if b in cleaned])
            for c in cols:
                if c.startswith("age_years"):
                    cleaned[c] = with_age[c].astype("float64")
                    added.append(c)
            steps.append(f"Datetime processing: parsed timestamps; derived {added} from birth date (raw birth date stays excluded)")
        by_role = self.schema.by_role()
        numeric_base = [c for c in by_role.get("numeric", []) if c in cleaned and c not in r.excluded and c != r.target] + added
        categorical = [c for c in by_role.get("categorical", []) + by_role.get("categorical_code", []) + by_role.get("boolean", [])
                       if c in cleaned and c not in r.excluded and c != r.target]
        pcfg = cfg["processing"]
        pre = FittedPreprocessor(PreprocessingConfig(time_features=False, rare_min_frequency=pcfg["rare_min_frequency"],
                                                     n_bins=pcfg["n_bins"], log_skew_threshold=pcfg["log_skew_threshold"]))
        pre.fit(cleaned[cleaned["_split"] == "train"], ColumnRoles(numeric=numeric_base, categorical=categorical, event_time=None))
        transformed, stats = pre.transform(cleaned)
        indicators = [c for c in transformed.columns if c.endswith("__was_missing")]
        numeric_features = [f"{c}{suffix}" for c in numeric_base for suffix in ("__robust", "__log", "__bin")
                            if f"{c}{suffix}" in transformed] + indicators
        logged = [c for c, p in pre.numeric.params.items() if p["log"]]
        steps += ["Fitted on training rows only: missing-value handling, rare-category grouping, numeric transforms "
                  f"(robust scaling + decile bins for all numeric features; log1p additionally for skewed: {logged or 'none'})",
                  "Numeric model inputs are the transformed columns (robust-scaled / log / decile bin), not the raw values",
                  "Absolute timestamps removed from model inputs: later periods lie outside the training range"]
        audit = {"cleaning": cleaning.summary, "audit_log": cleaning.audit.entries, "preprocessor": pre.to_dict(),
                 "transform_stats": stats, "quality_scores": quality.scores}
        return transformed, numeric_features, categorical, {"steps": steps, "audit": audit}

    def _build_e1(self) -> PreparedLevel:
        frame, numeric, categorical, extra = self._e1_frame()
        extra["steps"].append("Leakage checks: full feature-level scan not yet implemented (planned for Phase 4)")
        return self._finish("E1", frame, numeric + categorical, extra["steps"], extra)

    def _event_times(self, frame):
        times = parse_datetime_column(self.df.set_index("_row_id")[self.roles.time], "datetime")
        return frame["_row_id"].map(times)

    def _history_features(self) -> pd.DataFrame | None:
        if self._history is not None:
            return self._history
        r = self.roles
        base = pd.DataFrame({"_row_id": self.df["_row_id"], "entity": self.df[r.entity],
                             "time": parse_datetime_column(self.df[r.time], "datetime"),
                             "amount": pd.to_numeric(self.df[r.amount], errors="coerce")})
        if r.category:
            base["category"] = self.df[r.category].astype(str).str.strip()
        if r.merchant:
            base["merchant"] = self.df[r.merchant].astype(str).str.strip()
        base = base.dropna(subset=["time"])
        if base.empty:
            return None
        hc = HistoryConfig(entity="entity", time="time", amount="amount", category="category" if r.category else None,
                           merchant="merchant" if r.merchant else None, windows=tuple(self.cfg["features"]["windows"]))
        parts = [card_history(base, hc)]
        if r.merchant:
            parts.append(merchant_history(base, hc))
        hist = pd.concat(parts, axis=1)
        hist.insert(0, "_row_id", base["_row_id"].values)
        # point-in-time verification on a few entities: recomputes history from truncated data and compares
        ents = base["entity"].drop_duplicates().sample(min(10, base["entity"].nunique()), random_state=self.seed)
        sub = base[base["entity"].isin(ents)]
        self.pit = check_point_in_time(sub, lambda f: card_history(f, hc), "time", "entity", n_samples=min(100, len(sub)))
        self._history = hist
        return hist

    def _build_e2(self) -> PreparedLevel:
        frame, numeric, categorical, extra = self._e1_frame()
        r = self.roles
        temporal_cols = []
        if r.time:
            t = self._event_times(frame)
            hour, dow = t.dt.hour.astype("float64"), t.dt.dayofweek.astype("float64")
            temporal = {"hour": hour, "day_of_week": dow, "month": t.dt.month.astype("float64"),
                       "is_weekend": (dow >= 5).astype("float64"),
                       "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                       "day_of_week_sin": np.sin(2 * np.pi * dow / 7), "day_of_week_cos": np.cos(2 * np.pi * dow / 7)}
            for name, values in temporal.items():
                frame[name] = values
            temporal_cols = list(temporal)
            extra["steps"].append(f"Time features (raw and cyclical): {temporal_cols}")
        else:
            extra["steps"].append("Time features skipped: no datetime column selected")

        hist_cols = []
        if r.entity and r.time and r.amount:
            hist = self._history_features()
            if hist is not None:
                hist_cols = [c for c in HISTORY_FEATURES if c in hist]
                frame = frame.merge(hist[["_row_id"] + hist_cols], on="_row_id", how="left")
                extra["steps"].append(f"History features from strictly earlier transactions of the same {r.entity} "
                                      f"(no past labels): {len(hist_cols)} features")
                extra["steps"].append(f"Point-in-time check on real rows: {'passed' if self.pit['passed'] else 'FAILED'} "
                                      f"({self.pit['rows_checked']} rows)")
                extra["point_in_time"] = self.pit
            else:
                extra["steps"].append("History features skipped: no rows have a valid event time")
        else:
            missing = [name for name, v in [("entity", r.entity), ("time", r.time), ("amount", r.amount)] if not v]
            extra["steps"].append(f"History features skipped: no {', '.join(missing)} column selected")
        extra["steps"].append("Leakage checks: full feature-level scan not yet implemented (planned for Phase 4); "
                              "history features are verified point-in-time above")
        feats = numeric + temporal_cols + hist_cols + categorical
        return self._finish("E2", frame, feats, extra["steps"], extra)
