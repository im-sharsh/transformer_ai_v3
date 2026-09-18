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

from src.features.behavioral_features import (HistoryConfig, card_history, check_point_in_time, merchant_history,
                                              previous_transactions)
from src.ingestion.roles import DatasetRoles
from src.ingestion.schema_detector import SchemaReport
from src.preprocessing.cleaner import CleaningConfig, clean_dataset
from src.preprocessing.datetime_features import add_time_features
from src.preprocessing.pipeline import ColumnRoles, FittedPreprocessor, PreprocessingConfig
from src.preprocessing.sampling import SampleConfig, draw_sample, sample_report
from src.profiling.profiler import parse_datetime_column
from src.quality.leakage_detector import LeakageDetector
from src.quality.quality_engine import assess_quality

LEVELS = {"E0": "Raw", "E1": "Quality processed", "E2": "Feature engineered"}
# Only structural, unambiguous leaks are removed automatically. Strong predictive power alone (e.g. the amount)
# is flagged for human review, never removed: a legitimate signal can be very predictive.
AUTO_REMOVE_CHECKS = {"post_event_time", "target_word_in_text"}
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
                          "test": f"{q2} to {t.max()}", "method": "temporal (quantiles of event time)",
                          "cutoff": str(q1)}
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
            if n < 1 or n > len(frame):
                raise ValueError(f"Requested subset must be between 1 and {len(frame):,} rows; got {n:,}")
            # Start from the configured split fractions, then cap to the rows actually available in each split
            # and redistribute any deficit. Random splitting can be off by a few rows from the nominal fraction;
            # this guarantees the customer still receives exactly the row count they requested.
            requested = {"train": int(n * f_train), "validation": int(n * f_val)}
            requested["test"] = n - requested["train"] - requested["validation"]
            available = frame["split"].value_counts().to_dict()
            sizes = {k: min(requested[k], int(available.get(k, 0))) for k in ["train", "validation", "test"]}
            deficit = n - sum(sizes.values())
            while deficit > 0:
                progressed = False
                for k in ["train", "validation", "test"]:
                    capacity = int(available.get(k, 0)) - sizes[k]
                    if capacity <= 0:
                        continue
                    add = min(capacity, deficit)
                    sizes[k] += add
                    deficit -= add
                    progressed = True
                    if deficit == 0:
                        break
                if not progressed:
                    raise ValueError(f"Could not allocate the requested {n:,} rows across train/validation/test")
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
        extra = {"excluded": r.excluded}
        # Audit finding 3 (measured, not assumed): a full-precision timestamp string is near-unique per row, so
        # every value in TabularTokenizer's default quantile_bin mode falls below min_category_count and the
        # column collapses to the "unknown" token for ~100% of rows -- E0 effectively has *no* time signal at
        # all, not "raw" time signal. Opt-in (data.e0_add_time_epoch, default false until measured against the
        # baseline the same way findings 4/5 were): adds a numeric epoch-seconds column alongside the unchanged
        # raw string, so E0-vs-E1/E2 can be compared on processing *quality* rather than "has any time signal".
        if r.time and self.cfg.get("data", {}).get("e0_add_time_epoch", False):
            t = parse_datetime_column(rows[r.time], "datetime")
            epoch_col = f"{r.time}__epoch"
            rows = rows.copy()
            rows[epoch_col] = (t - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("float64")
            features = features + [epoch_col]
            steps.append(f"data.e0_add_time_epoch is on: added {epoch_col} (numeric epoch seconds) alongside "
                        f"the unchanged raw {r.time} string, so E0 has a usable time signal (see audit finding 3)")
        return self._finish("E0", rows, features, steps, extra)

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
        # Audit finding 1: __robust, __log and __bin are all monotonic (or near-monotonic) transforms of the same
        # raw value, and the tokenizer quantile-bins every "numeric" feature independently. Quantile binning is
        # invariant under monotonic transforms, so passing all three produced three redundant token positions
        # with zero net new information (verified empirically: __robust and __log re-binned match a direct bin
        # of the raw value 100% of the time). Only __robust is kept as the model input; __log and __bin are still
        # computed (available for future representation experiments, e.g. a continuous-value + coarse-bin input)
        # but are not selected here.
        numeric_features = [f"{c}__robust" for c in numeric_base if f"{c}__robust" in transformed] + indicators
        logged = [c for c, p in pre.numeric.params.items() if p["log"]]
        steps += ["Fitted on training rows only: missing-value handling, rare-category grouping, numeric transforms "
                  f"(robust scaling for all numeric features; log1p and decile bins also computed, for skewed "
                  f"columns: {logged or 'none'}, but not used as separate model inputs — redundant with the "
                  "robust-scaled value under quantile-bin tokenization)",
                  "Numeric model input is the robust-scaled value: one token per numeric column, not three",
                  "Absolute timestamps removed from model inputs: later periods lie outside the training range"]
        audit = {"cleaning": cleaning.summary, "audit_log": cleaning.audit.entries, "preprocessor": pre.to_dict(),
                 "transform_stats": stats, "quality_scores": quality.scores}
        return transformed, numeric_features, categorical, {"steps": steps, "audit": audit}

    def _run_leakage(self, level: str, frame: pd.DataFrame, features: list) -> tuple[list, list, str]:
        """Full feature-level leakage scan (Phase 4) on the level's final feature set. Needs a datetime column
        to hold out a later period; returns (findings, auto-removed features, step message)."""
        r = self.roles
        present = [c for c in features if c in frame]
        if not r.time:
            return [], [], "Leakage checks skipped: no datetime column selected"
        lcfg = self.cfg["leakage"]
        scan = frame[["_row_id", "_target"] + present].copy()
        scan["_event_time"] = self._event_times(frame)
        if r.entity:
            entity_of = self.df.set_index("_row_id")[r.entity]
            scan[r.entity] = scan["_row_id"].map(entity_of)
        det = LeakageDetector(target="_target", event_time="_event_time", entity=r.entity, roles=self.schema.by_role(),
                              birth_columns=r.birth, cutoff=self.split_info.get("boundaries", {}).get("cutoff"),
                              holdout_fraction=lcfg["holdout_fraction"], high_auc=lcfg["high_auc"],
                              medium_auc=lcfg["medium_auc"], instability_gap=lcfg["instability_gap"], seed=self.seed)
        report = det.detect(scan, dataset_id=level, features=present)
        findings = [f.__dict__ for f in report.findings]
        removed = sorted({f["feature"] for f in findings if f["leakage_risk"] == "high" and f["check"] in AUTO_REMOVE_CHECKS})
        msg = (f"Leakage checks: {len(findings)} findings on {len(present)} features (cutoff {report.split['cutoff']}, "
              f"{report.split['fit_rows']:,} fit rows / {report.split['holdout_rows']:,} later rows); "
              f"removed structural leaks: {removed or 'none'} (other findings are for human review)")
        return findings, removed, msg

    def _build_e1(self) -> PreparedLevel:
        frame, numeric, categorical, extra = self._e1_frame()
        findings, removed, msg = self._run_leakage("E1", frame, numeric + categorical)
        extra["steps"].append(msg)
        extra["leakage"] = findings
        kept = [c for c in numeric + categorical if c not in removed]
        return self._finish("E1", frame, kept, extra["steps"], extra)

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
        fcfg = self.cfg["features"]
        use_sequence = bool(fcfg.get("include_sequence_features", True))
        seq_len = int(fcfg.get("sequence_length", 3))
        hc = HistoryConfig(entity="entity", time="time", amount="amount", category="category" if r.category else None,
                           merchant="merchant" if r.merchant else None, windows=tuple(fcfg["windows"]),
                           sequence_length=seq_len)
        parts = [card_history(base, hc)]
        if r.merchant:
            parts.append(merchant_history(base, hc))
        if use_sequence:
            # Audit finding 5: these prevK_* columns (the entity's last K transactions) were already implemented
            # and leakage-tested but never reached E2 in the original pipeline. Wired in here because a separate
            # experiment (fine-tuning a language model on the equivalent text representation) found this to be
            # the single largest effect measured in that research: PR-AUC 0.87 vs 0.61-0.65 for aggregate-only
            # history, same seed, statistically significant. Point-in-time verified below like every other
            # history feature.
            parts.append(previous_transactions(base, hc))
        hist = pd.concat(parts, axis=1)
        hist.insert(0, "_row_id", base["_row_id"].values)
        # point-in-time verification on a few entities: recomputes history from truncated data and compares
        ents = base["entity"].drop_duplicates().sample(min(10, base["entity"].nunique()), random_state=self.seed)
        sub = base[base["entity"].isin(ents)]
        pit_card = check_point_in_time(sub, lambda f: card_history(f, hc), "time", "entity", n_samples=min(100, len(sub)))
        if use_sequence:
            pit_seq = check_point_in_time(sub, lambda f: previous_transactions(f, hc), "time", "entity",
                                          n_samples=min(100, len(sub)))
            self.pit = {"card_history": pit_card, "previous_transactions": pit_seq,
                       "passed": bool(pit_card["passed"] and pit_seq["passed"]), "rows_checked": pit_card["rows_checked"]}
        else:
            self.pit = {"card_history": pit_card, "passed": bool(pit_card["passed"]), "rows_checked": pit_card["rows_checked"]}
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

        hist_cols, seq_numeric, seq_categorical = [], [], []
        if r.entity and r.time and r.amount:
            hist = self._history_features()
            if hist is not None:
                hist_cols = [c for c in HISTORY_FEATURES if c in hist]
                fcfg = self.cfg["features"]
                if fcfg.get("include_sequence_features", True):
                    seq_len = int(fcfg.get("sequence_length", 3))
                    for k in range(1, seq_len + 1):
                        for suffix, numeric_kind in [("hours_ago", True), ("amount", True), ("hour", True),
                                                     ("category", False)]:
                            col = f"prev{k}_{suffix}"
                            if col in hist.columns:
                                (seq_numeric if numeric_kind else seq_categorical).append(col)
                merge_cols = hist_cols + seq_numeric + seq_categorical
                frame = frame.merge(hist[["_row_id"] + merge_cols], on="_row_id", how="left")
                seq_msg = (f" + {len(seq_numeric) + len(seq_categorical)} previous-transaction sequence features "
                          f"(last {int(self.cfg['features'].get('sequence_length', 3))})"
                          if seq_numeric or seq_categorical else "")
                extra["steps"].append(f"History features from strictly earlier transactions of the same {r.entity} "
                                      f"(no past labels): {len(hist_cols)} aggregate features{seq_msg}")
                extra["steps"].append(f"Point-in-time check on real rows: {'passed' if self.pit['passed'] else 'FAILED'} "
                                      f"({self.pit['rows_checked']} rows)")
                extra["point_in_time"] = self.pit
            else:
                extra["steps"].append("History features skipped: no rows have a valid event time")
        else:
            missing = [name for name, v in [("entity", r.entity), ("time", r.time), ("amount", r.amount)] if not v]
            extra["steps"].append(f"History features skipped: no {', '.join(missing)} column selected")
        feats = numeric + temporal_cols + hist_cols + seq_numeric + categorical + seq_categorical
        findings, removed, msg = self._run_leakage("E2", frame, feats)
        if hist_cols:
            msg += " (history features are additionally verified point-in-time above)"
        extra["steps"].append(msg)
        extra["leakage"] = findings
        return self._finish("E2", frame, [c for c in feats if c not in removed], extra["steps"], extra)
