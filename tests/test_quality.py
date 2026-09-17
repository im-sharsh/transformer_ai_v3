import numpy as np
import pandas as pd

from src.ingestion.roles import detect_roles, schema_for_profiling
from src.ingestion.schema_detector import detect_schema
from src.quality.quality_engine import assess_quality


def _check(report, name):
    matches = [c for c in report.checks if c.name == name]
    assert matches, f"check '{name}' not found; have {[c.name for c in report.checks]}"
    return matches[0]


def test_clean_transaction_data_scores_high(transactions):
    # The synthetic demo data deliberately includes a redundant `unix_time` column offset from the real
    # timestamp (mirroring a real-world quirk seen in public fraud datasets), so a systematic consistency
    # check is expected to fire here; this is not a bug. Everything else should be clean.
    schema = detect_schema(transactions, "tx")
    roles = detect_roles(transactions, schema)
    ps = schema_for_profiling(schema, roles, transactions)
    report = assess_quality(transactions, ps)
    assert report.scores["completeness"]["score"] == 100.0
    assert report.scores["uniqueness"]["score"] == 100.0
    assert report.records_with_issues == 0
    assert "epoch_vs_datetime:unix_time" in report.systematic_issues
    assert report.scores["overall"] > 85, report.scores


def test_quality_engine_finds_injected_problems(transactions):
    df = transactions.copy()
    df.loc[0:2, "trans_num"] = np.nan                              # 3 mandatory violations
    df.loc[10:13, "trans_num"] = df.loc[20, "trans_num"]           # duplicate ids
    df.loc[30:36, "amt"] = -1.0                                    # 7 negative amounts (amount-like column)
    df.loc[40:42, "trans_date_trans_time"] = "not-a-date"          # 3 invalid timestamps
    df.loc[50, "trans_date_trans_time"] = "2099-01-01 00:00:00"    # future timestamp

    schema = detect_schema(df, "dirty")
    roles = detect_roles(df, schema)
    ps = schema_for_profiling(schema, roles, df)
    report = assess_quality(df, ps)

    assert _check(report, "mandatory_fields").failed == 3
    assert _check(report, "duplicate_id:trans_num").failed == 5    # rows 10,11,12,13 and 20 share one id
    assert _check(report, "range:amt").failed == 7
    assert _check(report, "datetime_parse:trans_date_trans_time").failed == 3
    assert _check(report, "future_datetime:trans_date_trans_time").failed == 1
    assert report.scores["overall"] < 95


def test_quality_engine_is_generic_on_churn_dataset(churn):
    schema = detect_schema(churn, "churn")
    roles = detect_roles(churn, schema)
    ps = schema_for_profiling(schema, roles, churn)
    report = assess_quality(churn, ps)
    # No transaction-specific column names anywhere in this dataset; the engine must still run and score it.
    assert report.scores["overall"] is not None
    assert report.rows == len(churn)
    names = {c.name for c in report.checks}
    assert any(n.startswith("mandatory_fields") for n in names)


def test_structural_quality_checks_do_not_depend_on_column_names(transactions):
    """The role-based checks (completeness, duplicates, mandatory fields, datetime parsing, id format) must
    fire identically after every column is renamed. A few heuristic checks inherited from the original
    research code (amount sign, geographic coordinate range, birth-date plausibility) additionally use
    column-name hints and can lose coverage under renaming; this is a known, documented limitation
    (see PROJECT_STATUS.md), not something this test should hide by loosening its assertions."""
    renamed = transactions.rename(columns={c: f"col_{i}" for i, c in enumerate(transactions.columns)})
    schema = detect_schema(renamed, "renamed")
    roles = detect_roles(renamed, schema)
    report = assess_quality(renamed, schema_for_profiling(schema, roles, renamed))

    original_schema = detect_schema(transactions, "tx")
    original_roles = detect_roles(transactions, original_schema)
    original_report = assess_quality(transactions, schema_for_profiling(original_schema, original_roles, transactions))

    for dim in ("completeness", "uniqueness"):
        assert report.scores[dim]["score"] == original_report.scores[dim]["score"]
    role_based_prefixes = ("missing_values", "mandatory_fields", "duplicate_id", "exact_duplicate_rows",
                          "duplicate_rows_ignoring_ids", "datetime_parse", "future_datetime", "id_format",
                          "entity_attribute_stability", "target_values")
    original_role_checks = {c.name.split(":")[0]: c.failed for c in original_report.checks
                            if c.name.startswith(role_based_prefixes)}
    renamed_role_checks = {c.name.split(":")[0]: c.failed for c in report.checks if c.name.startswith(role_based_prefixes)}
    assert original_role_checks == renamed_role_checks, (original_role_checks, renamed_role_checks)
