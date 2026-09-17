import numpy as np
import pandas as pd

from src.ingestion.roles import detect_roles, schema_for_profiling
from src.ingestion.schema_detector import detect_schema
from src.preprocessing.basic_preprocessing import run_basic_preprocessing
from src.preprocessing.cleaner import CleaningConfig


def _prepare(df, dataset_id="ds"):
    schema = detect_schema(df, dataset_id)
    roles = detect_roles(df, schema)
    return schema_for_profiling(schema, roles, df), roles


def test_no_missing_values_reports_no_imputation_required(transactions):
    ps, roles = _prepare(transactions, "tx")
    result = run_basic_preprocessing(transactions, ps, roles)
    assert result.summary.missing_cells_before == 0
    assert "No imputation required" in result.summary.missing_values_action


def test_never_fabricates_duplicates_or_missing(transactions):
    ps, roles = _prepare(transactions, "tx")
    result = run_basic_preprocessing(transactions, ps, roles)
    assert result.summary.duplicates_detected == 0
    assert result.summary.duplicates_removed == 0
    assert result.summary.quarantined_rows == 0
    assert result.summary.rows_before == result.summary.rows_after == len(transactions)


def test_drops_redundant_and_identifier_columns(transactions):
    ps, roles = _prepare(transactions, "tx")
    result = run_basic_preprocessing(transactions, ps, roles)
    # unix_time is a systematic-consistency duplicate of the parsed timestamp; the row counter carries no signal.
    assert "unix_time" in result.summary.dropped_columns
    assert "Unnamed: 0" in result.summary.dropped_columns
    assert set(result.summary.identifier_columns_excluded) >= {"trans_num", "cc_num"}


def test_detects_and_removes_real_duplicates_generic_dataset(churn):
    df = pd.concat([churn, churn.iloc[:7]], ignore_index=True)     # 7 exact duplicate rows
    ps, roles = _prepare(df, "churn_dup")
    result = run_basic_preprocessing(df, ps, roles)
    assert result.summary.duplicates_detected == 7
    assert result.summary.duplicates_removed == 7
    assert result.summary.rows_after == result.summary.rows_before - 7


def test_normalizes_categorical_case_and_whitespace_variants(churn):
    ps, roles = _prepare(churn, "churn")
    result = run_basic_preprocessing(churn, ps, roles)
    assert result.summary.categorical_normalized > 0
    assert not result.cleaning.df["plan"].str.contains(r"\s$", regex=True).any()


def test_retain_policy_does_not_remove_duplicate_content(churn):
    # Duplicate content but a fresh identifier for each copy, so this exercises the duplicate-row policy alone
    # (a repeated primary-key value is a separate integrity issue, correctly quarantined regardless of this policy).
    extra = churn.iloc[:4].copy()
    extra["customer_id"] = churn["customer_id"].max() + 1 + range(len(extra))
    df = pd.concat([churn, extra], ignore_index=True)
    ps, roles = _prepare(df, "churn_retain")
    result = run_basic_preprocessing(df, ps, roles, config=CleaningConfig(duplicate_policy="retain"))
    assert result.summary.duplicates_detected == 4
    assert result.summary.duplicates_removed == 0
    assert result.summary.rows_after == result.summary.rows_before


def test_repeated_primary_key_is_quarantined_regardless_of_duplicate_policy(churn):
    dup_id = pd.concat([churn, churn.iloc[:3]], ignore_index=True)   # same customer_id AND same content
    ps, roles = _prepare(dup_id, "churn_dupid")
    result = run_basic_preprocessing(dup_id, ps, roles, config=CleaningConfig(duplicate_policy="retain"))
    assert result.summary.quarantined_rows > 0                       # conflicting identifier values, not a fabricated count


def test_summary_lines_are_well_formed_and_only_measured_values(transactions):
    ps, roles = _prepare(transactions, "tx")
    result = run_basic_preprocessing(transactions, ps, roles)
    lines = result.summary.to_lines()
    assert any("Processing time:" in l for l in lines)
    assert any("Not yet implemented" in l for l in lines)          # leakage checks are honestly reported as absent
    assert len(lines) >= 15


def test_cleaning_resolves_the_systematic_issue_it_detects(transactions):
    """The demo data has one deliberate systematic issue (a redundant, offset epoch column). Basic
    preprocessing should detect it via the quality engine, drop the redundant column, and a fresh quality
    assessment of the cleaned output should then score higher."""
    ps, roles = _prepare(transactions, "tx")
    before_score = result_before = None
    from src.quality.quality_engine import assess_quality
    before_score = assess_quality(transactions, ps).scores["overall"]
    result = run_basic_preprocessing(transactions, ps, roles)
    assert "unix_time" in result.summary.dropped_columns
    cleaned_content = result.cleaning.df.drop(columns=[c for c in result.cleaning.df.columns if c.startswith("_")])
    after_ps, after_roles = _prepare(cleaned_content, "tx_cleaned")
    after_score = assess_quality(cleaned_content, after_ps).scores["overall"]
    assert after_score > before_score


def test_generic_across_unrelated_dataset_no_hardcoded_names(churn):
    """No column from the transaction dataset (is_fraud, cc_num, trans_date_trans_time, amt) exists here."""
    assert not {"is_fraud", "cc_num", "trans_date_trans_time", "amt"} & set(churn.columns)
    ps, roles = _prepare(churn, "churn")
    result = run_basic_preprocessing(churn, ps, roles)
    assert result.summary.rows_before == len(churn)
    assert result.quality.scores["overall"] is not None
