import numpy as np
import pandas as pd
import pytest

from src.ingestion.roles import detect_roles, schema_for_profiling
from src.ingestion.schema_detector import detect_schema
from src.preprocessing.levels import DataPreparer, infer_roles
from src.utils.config import load_config

CFG = load_config()


def _prepare(df, dataset_id="ds", target=None):
    schema = detect_schema(df, dataset_id)
    roles = detect_roles(df, schema, target=target)
    return schema_for_profiling(schema, roles, df), roles


def test_infer_roles_resolves_amount_category_merchant_and_excludes_identifiers_pii_and_quasi_identifiers(transactions):
    ps, roles = _prepare(transactions, "tx")
    lr = infer_roles(transactions, ps, roles)
    assert (lr.target, lr.time, lr.entity) == ("is_fraud", "trans_date_trans_time", "cc_num")
    assert (lr.amount, lr.category, lr.merchant) == ("amt", "category", "merchant")
    for c in ["Unnamed: 0", "trans_num", "cc_num", "first", "last", "street", "dob", "zip", "lat", "long"]:
        assert c in lr.excluded
    assert "job" in lr.excluded, "job is unique per card: a quasi-identifier, even though it isn't flagged as PII"
    assert "state" not in lr.excluded and "gender" not in lr.excluded, "legitimate low-cardinality features are kept"


def test_infer_roles_requires_a_binary_target():
    df = pd.DataFrame({"x": range(30), "y": [0, 1, 2] * 10})
    ps, roles = _prepare(df, "multi", target="y")
    with pytest.raises(ValueError, match="binary target"):
        infer_roles(df, ps, roles)


def test_prepare_split_is_temporal_and_chronological(transactions):
    ps, roles = _prepare(transactions, "tx")
    lr = infer_roles(transactions, ps, roles)
    dp = DataPreparer(transactions, ps, lr, CFG, rows=2000, seed=42)
    info = dp.prepare_split()
    assert info["boundaries"]["method"].startswith("temporal")
    rep = info["sample"]
    assert sum(rep[s]["rows"] for s in ["train", "validation", "test"]) <= 2000


def test_prepare_split_falls_back_to_random_without_a_datetime_column(transactions):
    df = transactions.drop(columns=["trans_date_trans_time"])
    ps, roles = _prepare(df, "tx_nodt")
    lr = infer_roles(df, ps, roles)
    assert lr.time is None
    dp = DataPreparer(df, ps, lr, CFG, rows=500, seed=42)
    info = dp.prepare_split()
    assert info["boundaries"]["method"] == "random (no datetime column selected)"


def test_prepare_split_is_reproducible_for_the_same_seed(transactions):
    ps, roles = _prepare(transactions, "tx")
    lr = infer_roles(transactions, ps, roles)
    dp1 = DataPreparer(transactions, ps, lr, CFG, rows=1000, seed=7)
    dp2 = DataPreparer(transactions, ps, lr, CFG, rows=1000, seed=7)
    dp1.prepare_split()
    dp2.prepare_split()
    pd.testing.assert_frame_equal(dp1.sample_ids.reset_index(drop=True), dp2.sample_ids.reset_index(drop=True))


def test_e0_keeps_raw_values_and_the_shared_exclusion_policy(transactions):
    ps, roles = _prepare(transactions, "tx")
    lr = infer_roles(transactions, ps, roles)
    dp = DataPreparer(transactions, ps, lr, CFG, rows=1000, seed=42)
    dp.prepare_split()
    e0 = dp.build("E0")
    assert "job" not in e0.features and "cc_num" not in e0.features
    assert "amt" in e0.features and "unix_time" in e0.features   # E0 keeps even the redundant raw epoch column
    train = e0.frames["train"]
    original = transactions.reset_index(drop=True)
    row_id = train["_row_id"].iloc[0]
    idx = int(row_id[1:])
    assert train.loc[train["_row_id"] == row_id, "amt"].iloc[0] == original.loc[idx, "amt"]


def test_e1_transforms_numeric_features_instead_of_using_raw_values(transactions):
    ps, roles = _prepare(transactions, "tx")
    lr = infer_roles(transactions, ps, roles)
    dp = DataPreparer(transactions, ps, lr, CFG, rows=2000, seed=42)
    dp.prepare_split()
    e1 = dp.build("E1")
    assert "amt" not in e1.features
    assert {"amt__robust", "amt__log", "amt__bin"} <= set(e1.features)
    assert not any(c.startswith("trans_date_trans_time") for c in e1.features), "no absolute timestamp in E1 inputs"
    assert "unix_time" not in e1.features, "detected as a redundant systematic duplicate and dropped by cleaning"
    assert any("Phase 4" in s for s in e1.info["steps"])   # leakage checks honestly reported as not yet implemented


def test_e1_adds_a_missing_value_indicator_fitted_on_train_rows_only(transactions):
    df = transactions.copy()
    df.loc[df.index[:50], "amt"] = np.nan
    ps, roles = _prepare(df, "tx_missing")
    lr = infer_roles(df, ps, roles)
    dp = DataPreparer(df, ps, lr, CFG, rows="full", seed=42)
    dp.prepare_split()
    e1 = dp.build("E1")
    assert "amt__was_missing" in e1.features
    assert e1.frames["train"]["amt__was_missing"].sum() > 0


def test_e0_vs_e1_shows_the_researched_contrast_in_column_treatment(transactions):
    """The whole point of E0 vs E1 vs E2 is a measurable contrast: E0 is the uncurated baseline."""
    ps, roles = _prepare(transactions, "tx")
    lr = infer_roles(transactions, ps, roles)
    dp = DataPreparer(transactions, ps, lr, CFG, rows=2000, seed=42)
    dp.prepare_split()
    assert "unix_time" in dp.build("E0").features
    assert "unix_time" not in dp.build("E1").features


def test_e2_adds_cyclical_time_and_point_in_time_safe_history_features(transactions):
    ps, roles = _prepare(transactions, "tx")
    lr = infer_roles(transactions, ps, roles)
    dp = DataPreparer(transactions, ps, lr, CFG, rows=3000, seed=42)
    dp.prepare_split()
    e2 = dp.build("E2")
    for c in ["hour", "hour_sin", "hour_cos", "day_of_week", "day_of_week_sin", "day_of_week_cos", "is_weekend"]:
        assert c in e2.features
    assert "card_amount_mean_before" in e2.features
    assert "card_txn_count_before" not in e2.features, "diagnostic-only (grows with calendar time): not a model input"
    pit = e2.info["point_in_time"]
    assert pit["passed"], pit["examples"]


def test_e2_gracefully_skips_history_features_without_entity_or_amount(churn):
    ps, roles = _prepare(churn, "churn")
    lr = infer_roles(churn, ps, roles)
    assert lr.entity is None and lr.amount is None
    dp = DataPreparer(churn, ps, lr, CFG, rows=1000, seed=42)
    dp.prepare_split()
    e2 = dp.build("E2")
    assert any("History features skipped" in s for s in e2.info["steps"])
    assert "point_in_time" not in e2.info
    assert "hour" in e2.features                        # time features still built: signup_date exists


def test_e2_skips_time_features_gracefully_without_a_datetime_column(churn):
    df = churn.drop(columns=["signup_date"])
    ps, roles = _prepare(df, "churn_nodt")
    lr = infer_roles(df, ps, roles)
    assert lr.time is None
    dp = DataPreparer(df, ps, lr, CFG, rows=1000, seed=42)
    dp.prepare_split()
    e2 = dp.build("E2")
    assert any("Time features skipped" in s for s in e2.info["steps"])
    assert "hour" not in e2.features


def test_build_caches_the_prepared_level(transactions):
    ps, roles = _prepare(transactions, "tx")
    lr = infer_roles(transactions, ps, roles)
    dp = DataPreparer(transactions, ps, lr, CFG, rows=500, seed=42)
    dp.prepare_split()
    assert dp.build("E1") is dp.build("E1")
