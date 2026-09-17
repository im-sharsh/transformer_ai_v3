"""Basic preprocessing layer (generic, dataset-independent).

Sits between Quality Analysis and E0/E1/E2 in the architecture:

    Quality Analysis -> Basic Preprocessing -> E0 / E1 / E2

Everything here is *stateless*: no statistic is fitted on the data (no imputation value, no learned
vocabulary, no scaler), so it is safe to run on the whole dataset before any train/validation/test split
exists. Train-fitted steps (missing-value imputation values, rare-category grouping, numeric scaling) are
a separate, later stage that needs the split from Phase 3 (see src/preprocessing/pipeline.py, not wired in
yet; see _pending/README.md).

What this stage does, using the existing quality engine and cleaner:
  - type normalization (numeric/datetime parsing)
  - duplicate detection and a configurable policy (report-only by default)
  - invalid / non-finite value handling (values that fail a quality check become missing, never fabricated)
  - categorical normalization (whitespace, case variants)
  - datetime parsing and validity
  - identifier handling (dropped from the preprocessing report's "usable columns", never coerced to numeric)
  - a summary report with only measured values (see PreprocessingSummary)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.ingestion.roles import DatasetRoles
from src.ingestion.schema_detector import SchemaReport
from src.preprocessing.cleaner import CleaningConfig, CleaningResult, clean_dataset
from src.quality.quality_engine import QualityReport, assess_quality

INTERNAL_COLUMNS = ["_row_id", "_protected", "_record_status", "_flags"]


@dataclass
class PreprocessingSummary:
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    dropped_columns: list
    missing_cells_before: int
    missing_values_action: str
    duplicates_detected: int
    duplicates_removed: int
    duplicates_retained: int
    duplicate_policy: str
    quarantined_rows: int
    invalid_values_repaired: int
    numerical_columns: int
    categorical_columns: int
    categorical_normalized: int
    datetime_columns_parsed: int
    datetime_invalid_values: int
    identifier_columns_excluded: list
    non_finite_values_remaining: int
    potential_leakage_checks: str
    processing_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_lines(self) -> list[str]:
        return [
            f"Rows before:                {self.rows_before:,}",
            f"Rows after:                  {self.rows_after:,}",
            f"Columns before:              {self.columns_before:,}",
            f"Columns after:               {self.columns_after:,}",
            f"Missing cells (before):      {self.missing_cells_before:,}",
            f"Missing value handling:      {self.missing_values_action}",
            f"Duplicates detected:         {self.duplicates_detected:,}",
            f"Duplicates removed:          {self.duplicates_removed:,} (policy: {self.duplicate_policy})",
            f"Duplicates retained:         {self.duplicates_retained:,}",
            f"Quarantined rows:            {self.quarantined_rows:,}",
            f"Invalid values repaired:     {self.invalid_values_repaired:,}",
            f"Numerical columns:           {self.numerical_columns:,}",
            f"Categorical columns:         {self.categorical_columns:,}",
            f"Categorical values normalized: {self.categorical_normalized:,}",
            f"Datetime columns parsed:     {self.datetime_columns_parsed:,} "
            f"({self.datetime_invalid_values:,} invalid values found)",
            f"Identifier columns excluded: {len(self.identifier_columns_excluded):,}",
            f"Non-finite values remaining: {self.non_finite_values_remaining:,}",
            f"Potential leakage checks:    {self.potential_leakage_checks}",
            f"Processing time:             {self.processing_seconds:.2f} s",
        ]


@dataclass
class BasicPreprocessingResult:
    quality: QualityReport
    cleaning: CleaningResult
    summary: PreprocessingSummary


def _missing_action(before: int, config: CleaningConfig) -> str:
    if before == 0:
        return "No imputation required (0 missing cells detected)"
    return ("Invalid or unparseable values are set to missing and reported; value imputation is fitted on "
           "training rows only and runs during E1 / E2 processing (Phase 3), not at this stage")


def run_basic_preprocessing(df: pd.DataFrame, schema: SchemaReport, roles: DatasetRoles,
                            config: CleaningConfig | None = None) -> BasicPreprocessingResult:
    """Runs quality assessment, then stateless cleaning, on the whole dataset (no split exists yet)."""
    t0 = time.time()
    cfg = config or CleaningConfig()
    missing_before = int(df.isna().sum().sum())

    quality = assess_quality(df, schema)
    cleaning = clean_dataset(df, schema, quality, protected=None, config=cfg)
    out = cleaning.df

    by_role = schema.by_role()
    numeric_cols = [c for c, r in roles.columns.items() if r == "NUMERICAL" and c in out]
    categorical_cols = [c for c, r in roles.columns.items() if r == "CATEGORICAL" and c in out]
    datetime_cols = by_role.get("datetime", []) + by_role.get("datetime_epoch", [])

    def audit_total(operation: str, key: str = "records_modified") -> int:
        return sum(e[key] for e in cleaning.audit.entries if e["operation"] == operation)

    dup_entry = next((e for e in cleaning.audit.entries if e["operation"] == "duplicates"), None)
    duplicates_found = dup_entry["details"]["duplicates_found"] if dup_entry else 0
    invalid_repaired = audit_total("invalid_to_null")
    categorical_normalized = audit_total("normalize_whitespace") + audit_total("merge_case_variants")
    datetime_invalid = audit_total("convert_datetime")

    finite_check_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(out[c])]
    non_finite = 0
    for c in finite_check_cols:
        x = pd.to_numeric(out[c], errors="coerce")
        non_finite += int((np.isinf(x)).sum())

    summary = PreprocessingSummary(
        rows_before=cleaning.summary["rows_in"], rows_after=cleaning.summary["rows_out"],
        columns_before=cleaning.summary["columns_in"], columns_after=cleaning.summary["columns_out"],
        dropped_columns=cleaning.summary["dropped_columns"], missing_cells_before=missing_before,
        missing_values_action=_missing_action(missing_before, cfg),
        duplicates_detected=duplicates_found, duplicates_removed=cleaning.summary["duplicates_removed"],
        duplicates_retained=duplicates_found - cleaning.summary["duplicates_removed"], duplicate_policy=cfg.duplicate_policy,
        quarantined_rows=cleaning.summary["rows_quarantined"], invalid_values_repaired=invalid_repaired,
        numerical_columns=len(numeric_cols), categorical_columns=len(categorical_cols),
        categorical_normalized=categorical_normalized, datetime_columns_parsed=len(datetime_cols),
        datetime_invalid_values=datetime_invalid, identifier_columns_excluded=sorted(roles.identifiers),
        non_finite_values_remaining=non_finite,
        potential_leakage_checks="Not yet implemented (planned for Phase 4: leakage detector)",
        processing_seconds=round(time.time() - t0, 2))
    return BasicPreprocessingResult(quality=quality, cleaning=cleaning, summary=summary)
