"""Stateless (row-wise) cleaning.

Nothing here learns statistics from the data, so it is safe to run before the
train/validation/test split. Statistic-based steps (imputation, rare-category
grouping, scaling) run after the split and are fitted on training rows only.

Protected rows (validation/test periods) are never removed or quarantined.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.ingestion.schema_detector import SchemaReport
from src.utils.audit import AuditLog, ChangeLog
from src.preprocessing import duplicates as D
from src.profiling.profiler import parse_datetime_column
from src.quality.quality_engine import QualityReport

logger = logging.getLogger("CLEANING")

INTERNAL = ["_row_id", "_source_file", "_protected", "_record_status", "_flags"]
STATUS_ORDER = ["valid", "repaired", "invalid", "quarantine"]


@dataclass
class CleaningConfig:
    drop_systematic_redundant_columns: bool = True
    drop_row_index: bool = True
    drop_constant_columns: bool = True
    convert_types: bool = True
    zero_pad_codes: bool = True
    normalize_whitespace: bool = True
    merge_case_variants: bool = True
    strip_shared_prefixes: bool = True
    invalid_value_policy: str = "nullify"          # nullify | quarantine
    quarantine_missing_mandatory: bool = True
    duplicate_policy: str = "remove"               # remove | quarantine | retain
    repeated_transaction_policy: str = "flag"      # flag | remove

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CleaningResult:
    df: pd.DataFrame
    quarantine: pd.DataFrame
    removed_duplicates: pd.DataFrame
    audit: AuditLog
    changes: ChangeLog
    summary: dict = field(default_factory=dict)


class Cleaner:
    def __init__(self, config: CleaningConfig | None = None):
        self.cfg = config or CleaningConfig()

    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame, schema: SchemaReport, quality: QualityReport | None = None,
              protected: pd.Series | None = None) -> CleaningResult:
        t0 = time.time()
        cfg, audit, changes = self.cfg, AuditLog(), ChangeLog()
        rows_in, cols_in = df.shape
        logger.info("START cleaning %s: %d rows x %d columns", schema.dataset_id, rows_in, cols_in)

        df = df.copy()                                         # raw input is never modified
        if "_row_id" not in df:
            df["_row_id"] = [f"{schema.dataset_id}:{i}" for i in range(len(df))]
        df["_protected"] = (protected if protected is not None else pd.Series(False, index=df.index)).astype(bool)
        status = pd.Series(0, index=df.index, dtype="int8")    # index into STATUS_ORDER
        flags: dict[str, pd.Series] = {}
        quarantine_reasons: dict[str, pd.Series] = {}
        cols = {c: cs for c, cs in schema.columns.items() if c in df}
        roles = {r: [c for c in cs if c in df] for r, cs in schema.by_role().items()}
        event_time = next((c for c in roles.get("datetime", []) if cols[c].pii_type != "birth_date"), None)
        birth_cols = [c for c in roles.get("datetime", []) if cols[c].pii_type == "birth_date"]

        def bump(mask, level):
            status.loc[mask[mask].index] = np.maximum(status.loc[mask[mask].index], STATUS_ORDER.index(level))

        # 1. Drop columns that carry no usable information --------------------------
        before = df.shape[1]
        drops = {}
        if cfg.drop_systematic_redundant_columns and quality is not None:
            for chk in quality.checks:
                if chk.systematic and chk.name.startswith("epoch_vs_datetime:"):
                    col = chk.name.split(":", 1)[1]
                    drops[col] = (f"systematic consistency failure ({chk.failed:,} rows disagree with "
                                  f"{chk.columns[1]}; median offset {chk.details.get('median_offset_days')} days) "
                                  f"and redundant with {chk.columns[1]}")
        if cfg.drop_row_index:
            for c in roles.get("row_index", []):
                drops[c] = "row counter from the export; replaced by _row_id"
        if cfg.drop_constant_columns:
            for c in roles.get("constant", []):
                drops[c] = "constant column"
        if drops:
            df = df.drop(columns=list(drops))
            for c in drops:
                cols.pop(c, None)
            audit.add("clean", "drop_columns", "; ".join(f"{c}: {r}" for c, r in drops.items()),
                      len(df), len(df), before, df.shape[1], dropped=list(drops))
            logger.info("Dropped columns %s", list(drops))

        # 2. Type conversion ----------------------------------------------------------
        if cfg.convert_types:
            for c in [c for c in roles.get("datetime", []) if c in df]:
                original = df[c]
                parsed = parse_datetime_column(original, "datetime")
                failed = parsed.isna() & original.notna()
                df[c] = parsed
                self._log_bulk(audit, "convert_datetime", c, len(df), df.shape[1], failed,
                               "text timestamps parsed to datetime; unparseable values become NaT")
                if failed.any():
                    changes.add(df.loc[failed, "_row_id"], c, original[failed], parsed[failed],
                                "parse_failure_to_null", "value is not a valid datetime", cols[c].pii_type)
                    flags[f"invalid_datetime:{c}"] = failed
                    bump(failed, "invalid")
            for c in [c for c in roles.get("numeric", []) if c in df and not pd.api.types.is_numeric_dtype(df[c])]:
                original = df[c]
                parsed = pd.to_numeric(original, errors="coerce")
                failed = parsed.isna() & original.notna()
                df[c] = parsed
                self._log_bulk(audit, "convert_numeric", c, len(df), df.shape[1], failed,
                               "numbers stored as text converted; unparseable values become NaN")
                if failed.any():
                    changes.add(df.loc[failed, "_row_id"], c, original[failed], parsed[failed],
                                "parse_failure_to_null", "value is not a valid number", cols[c].pii_type)
                    flags[f"invalid_number:{c}"] = failed
                    bump(failed, "invalid")

        # 3. Code columns: restore leading zeros lost by numeric storage ---------------
        if cfg.zero_pad_codes:
            for c in [c for c in roles.get("categorical_code", []) if c in df]:
                text = df[c].astype("string")
                text = text.str.replace(r"\.0$", "", regex=True)
                digits = text.str.fullmatch(r"\d+").fillna(False)
                lengths = text[digits].str.len()
                if lengths.empty:
                    continue
                modal = int(lengths.mode().iloc[0])
                short = digits & (text.str.len() < modal)
                if (lengths == modal).mean() >= 0.5 and short.any():
                    padded = text.where(~short, text.str.zfill(modal))
                    changes.add(df.loc[short, "_row_id"], c, text[short], padded[short],
                                "zero_pad_code", f"code stored as number lost leading zeros (width {modal})",
                                cols[c].pii_type)
                    df[c] = padded
                    bump(short, "repaired")
                    audit.add("clean", "zero_pad_code", f"{c}: restored leading zeros to width {modal}",
                              len(df), len(df), df.shape[1], df.shape[1], records_modified=int(short.sum()))
                else:
                    df[c] = text
                    audit.add("clean", "code_as_text", f"{c}: stored as text so it is never used arithmetically",
                              len(df), len(df), df.shape[1], df.shape[1])

        # 4. Categorical text normalisation -------------------------------------------
        text_cols = [c for c in roles.get("categorical", []) + roles.get("boolean", []) if c in df
                     and not pd.api.types.is_numeric_dtype(df[c])]
        for c in text_cols:
            s = df[c]
            if cfg.normalize_whitespace:
                stripped = s.str.strip().str.replace(r"\s{2,}", " ", regex=True)
                changed = s.notna() & s.ne(stripped)
                if changed.any():
                    changes.add(df.loc[changed, "_row_id"], c, s[changed], stripped[changed],
                                "normalize_whitespace", "leading/trailing or repeated spaces", cols[c].pii_type)
                    bump(changed, "repaired")
                    audit.add("clean", "normalize_whitespace", c, len(df), len(df), df.shape[1], df.shape[1],
                              records_modified=int(changed.sum()))
                df[c] = s = stripped
            if cfg.merge_case_variants:
                counts = s.value_counts()
                keys = pd.Series(counts.index.str.casefold(), index=counts.index)
                canonical = counts.groupby(keys.values).idxmax()          # most frequent spelling wins
                mapping = {v: canonical[k] for v, k in keys.items() if canonical[k] != v}
                if mapping:
                    changed = s.isin(list(mapping))
                    merged = s.where(~changed, s.map(mapping))
                    changes.add(df.loc[changed, "_row_id"], c, s[changed], merged[changed],
                                "merge_case_variant", "same category written with different capitalisation",
                                cols[c].pii_type)
                    df[c] = merged
                    bump(changed, "repaired")
                    audit.add("clean", "merge_case_variants", c, len(df), len(df), df.shape[1], df.shape[1],
                              records_modified=int(changed.sum()), variants_merged=len(mapping))
            if cfg.strip_shared_prefixes:
                prefixes = [re.match(r"shared_prefix:'(.*)'$", f).group(1) for f in cols[c].flags
                            if f.startswith("shared_prefix:")]
                for prefix in prefixes:
                    has = df[c].str.startswith(prefix).fillna(False)
                    new = df[c].where(~has, df[c].str.slice(len(prefix)))
                    changes.add(df.loc[has, "_row_id"], c, df.loc[has, c], new[has], "strip_shared_prefix",
                                f"prefix '{prefix}' is shared by (almost) all values and carries no information",
                                cols[c].pii_type)
                    df[c] = new
                    audit.add("clean", "strip_shared_prefix", f"{c}: removed '{prefix}' (column-level change)",
                              len(df), len(df), df.shape[1], df.shape[1], records_modified=int(has.sum()),
                              target_related_word="prefix_contains_target_related_word" in cols[c].flags)

        # 5. Invalid values found by the quality engine ---------------------------------
        if quality is not None:
            for chk in quality.checks:
                if chk.failed == 0 or chk.systematic or chk.mask is None:
                    continue
                kind = chk.name.split(":", 1)[0]
                mask = chk.mask.reindex(df.index, fill_value=False)
                if kind in ("range", "allowed_values") or (kind == "plausible_age"):
                    col = chk.columns[0]
                    if col not in df:
                        continue
                    if cfg.invalid_value_policy == "quarantine":
                        quarantine_reasons[chk.name] = mask
                    else:
                        changes.add(df.loc[mask, "_row_id"], col, df.loc[mask, col],
                                    pd.Series(np.nan, index=mask[mask].index), "invalid_to_null",
                                    f"failed {chk.name}", cols[col].pii_type if col in cols else None)
                        df.loc[mask, col] = pd.NaT if pd.api.types.is_datetime64_any_dtype(df[col]) else np.nan
                        flags[chk.name] = mask
                        bump(mask, "invalid")
                        audit.add("clean", "invalid_to_null", f"{chk.name}: impossible values set to missing",
                                  len(df), len(df), df.shape[1], df.shape[1], records_modified=int(mask.sum()))
                elif kind == "order":
                    earlier = chk.columns[0]
                    if earlier in birth_cols:                      # a birth date after the event is the wrong value
                        changes.add(df.loc[mask, "_row_id"], earlier, df.loc[mask, earlier],
                                    pd.Series(pd.NaT, index=mask[mask].index), "invalid_to_null",
                                    f"failed {chk.name}", cols[earlier].pii_type)
                        df.loc[mask, earlier] = pd.NaT
                        bump(mask, "invalid")
                    flags[chk.name] = mask
                elif kind == "future_datetime" and chk.columns[0] == event_time:
                    quarantine_reasons[chk.name] = mask
                elif kind in ("id_format", "entity_attribute_stability", "foreign_key"):
                    flags[chk.name] = mask                         # informative; value kept as-is
                elif kind == "repeated_transactions":
                    if cfg.repeated_transaction_policy == "remove":
                        quarantine_reasons[chk.name] = mask
                    else:
                        flags[chk.name] = mask

        # 6. Records that cannot be used at all --------------------------------------
        if cfg.quarantine_missing_mandatory:
            mandatory = [c for c in [schema.target, *roles.get("record_id", []),
                                     *(roles.get("entity_id", [])[:1]), event_time] if c and c in df]
            for c in mandatory:
                missing = df[c].isna()
                if c == schema.target:
                    missing |= ~pd.to_numeric(df[c], errors="coerce").isin([0, 1])
                if missing.any():
                    quarantine_reasons[f"missing_mandatory:{c}"] = missing

        # 7. Duplicates ---------------------------------------------------------------
        content_cols = [c for c in df.columns if c not in INTERNAL]
        id_cols = [c for c in roles.get("record_id", []) if c in df]
        rows_before_dupes = len(df)
        exact = D.find_duplicates(df, content_cols)
        ignoring_ids = D.find_duplicates(df, [c for c in content_cols if c not in id_cols]) & ~exact
        dup_mask = exact | ignoring_ids
        removed_duplicates = df.iloc[0:0].copy()
        if dup_mask.any() and cfg.duplicate_policy != "retain":
            removable = dup_mask & ~df["_protected"]
            flags["duplicate_of_earlier_row"] = dup_mask & df["_protected"]
            if cfg.duplicate_policy == "remove":
                removed_duplicates = df[removable].assign(_removal_reason=np.where(
                    exact[removable], "exact_duplicate", "duplicate_ignoring_ids"))
                df = df[~removable]
                status = status[~removable]
                flags = {k: v[~removable] for k, v in flags.items()}
                quarantine_reasons = {k: v[~removable] for k, v in quarantine_reasons.items()}
            else:
                quarantine_reasons["duplicate"] = removable
        elif dup_mask.any():
            flags["duplicate_of_earlier_row"] = dup_mask
        for c in id_cols:
            conflict = D.conflicting_ids(df, c)
            if conflict.any():
                quarantine_reasons[f"conflicting_id:{c}"] = conflict
        audit.add("clean", "duplicates", f"policy={cfg.duplicate_policy}; protected rows only flagged",
                  rows_before_dupes, len(df), df.shape[1], df.shape[1],
                  records_removed=len(removed_duplicates), exact_duplicates=int(exact.sum()),
                  duplicates_ignoring_ids=int(ignoring_ids.sum()), duplicates_removed=len(removed_duplicates),
                  duplicates_found=int(dup_mask.sum()))

        # 8. Quarantine (never for protected rows) and final status ------------------
        reason_text = pd.Series("", index=df.index, dtype="object")
        any_reason = pd.Series(False, index=df.index)
        for name, m in quarantine_reasons.items():
            m = m.reindex(df.index, fill_value=False)
            reason_text.loc[m] += name + ";"
            any_reason |= m
        to_quarantine = any_reason & ~df["_protected"]
        kept_despite = any_reason & df["_protected"]
        if kept_despite.any():
            flags["protected_row_kept_despite_problem"] = kept_despite
            bump(kept_despite, "invalid")
        bump(to_quarantine, "quarantine")

        flag_text = pd.Series("", index=df.index, dtype="object")
        for name, m in flags.items():
            m = m.reindex(df.index, fill_value=False)
            flag_text.loc[m] += name + ";"
        flag_text.loc[kept_despite] += reason_text[kept_despite]
        df["_record_status"] = status.map(dict(enumerate(STATUS_ORDER)))
        df["_flags"] = flag_text.str.rstrip(";")

        quarantine = df[to_quarantine].assign(_quarantine_reason=reason_text[to_quarantine].str.rstrip(";"))
        df = df[~to_quarantine]
        audit.add("clean", "quarantine", "unusable records moved to quarantine (protected rows kept and flagged)",
                  len(df) + len(quarantine), len(df), df.shape[1], df.shape[1],
                  records_removed=len(quarantine), protected_kept_with_problems=int(kept_despite.sum()))

        status_counts = df["_record_status"].value_counts().to_dict()
        status_counts["quarantine"] = len(quarantine)
        summary = {
            "dataset_id": schema.dataset_id, "rows_in": rows_in, "rows_out": len(df),
            "rows_quarantined": len(quarantine), "duplicates_removed": len(removed_duplicates),
            "rows_reconcile": rows_in == len(df) + len(quarantine) + len(removed_duplicates),
            "columns_in": cols_in, "columns_out": len([c for c in df.columns if c not in INTERNAL]),
            "dropped_columns": list(drops), "record_status": {k: int(v) for k, v in status_counts.items()},
            "flag_counts": {k: int(v.reindex(df.index, fill_value=False).sum()) for k, v in flags.items()},
            "quarantine_reasons": ({k: int(v) for k, v in quarantine["_quarantine_reason"].str.split(";")
                                   .explode().value_counts().items()} if len(quarantine) else {}),
            "protected_rows": int(df["_protected"].sum()), "seconds": round(time.time() - t0, 1),
            "config": cfg.to_dict(),
        }
        assert summary["rows_reconcile"], "row counts do not reconcile"
        logger.info("Cleaned: %d -> %d rows (%d quarantined, %d duplicates removed) in %.1fs",
                    rows_in, len(df), len(quarantine), len(removed_duplicates), summary["seconds"])
        return CleaningResult(df=df, quarantine=quarantine, removed_duplicates=removed_duplicates,
                              audit=audit, changes=changes, summary=summary)

    @staticmethod
    def _log_bulk(audit, operation, column, rows, ncols, failed, reason):
        audit.add("clean", operation, f"{column}: {reason}", rows, rows, ncols, ncols,
                  records_modified=int(failed.sum()))


def clean_dataset(df, schema, quality=None, protected=None, config: CleaningConfig | None = None) -> CleaningResult:
    return Cleaner(config).clean(df, schema, quality, protected)


def save_cleaning_result(result: CleaningResult, project_root, inputs: list[dict],
                         quality_before: dict | None = None, quality_after: dict | None = None,
                         pipeline_version: str = "0.1.0", extra: dict | None = None) -> dict:
    """Write a new, never-overwritten dataset version plus its manifest, logs and quarantine."""
    from pathlib import Path
    from src.utils.audit import next_version

    project_root = Path(project_root)
    version = next_version(project_root / "data" / "processed")
    out = project_root / "data" / "processed" / version
    q_out = project_root / "data" / "quarantine" / version
    out.mkdir(parents=True, exist_ok=False)
    q_out.mkdir(parents=True, exist_ok=True)

    files = {"cleaned": out / "cleaned.parquet"}
    result.df.to_parquet(files["cleaned"], index=False)
    if len(result.quarantine):
        files["quarantine"] = q_out / "quarantine.parquet"
        result.quarantine.to_parquet(files["quarantine"], index=False)
    if len(result.removed_duplicates):
        files["removed_duplicates"] = q_out / "removed_duplicates.parquet"
        result.removed_duplicates.to_parquet(files["removed_duplicates"], index=False)
    result.audit.save(out)
    result.changes.save(out)

    manifest = {
        "dataset_version": version, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline_version": pipeline_version, "stage": "stateless_cleaning", "inputs": inputs,
        "summary": result.summary, "quality_before": quality_before, "quality_after": quality_after,
        "files": {k: str(v.relative_to(project_root)) for k, v in files.items()}, **(extra or {}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Saved %s to %s", version, out)
    return manifest
