import pandas as pd

from src.profiling.feature_profile import profile_feature_support, recommend_feature_profile


def test_repeated_entity_data_recommends_behavioral_e2():
    rows = []
    for entity in range(50):
        for k in range(8):
            rows.append({
                "entity": f"C{entity}",
                "time": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=k),
                "amount": 10.0 + k,
                "merchant": f"M{k % 5}",
                "category": "x",
            })
    df = pd.DataFrame(rows)
    p = profile_feature_support(df, entity="entity", time="time", amount="amount",
                                counterparty="merchant", category="category", sequence_length=3)
    r = recommend_feature_profile(p)
    assert p.entity_repeat_row_share == 1.0
    assert p.entity_sequence_coverage > 0.5
    assert r.level == "E2"
    assert "history" in r.feature_groups
    assert "sequence" in r.feature_groups


def test_sparse_origin_data_falls_back_to_continuous_e1():
    n = 500
    df = pd.DataFrame({
        "entity": [f"C{i}" for i in range(n)],
        "time": pd.date_range("2024-01-01", periods=n, freq="min"),
        "amount": 1.0,
        "merchant": [f"M{i % 20}" for i in range(n)],
        "category": "PAYMENT",
    })
    p = profile_feature_support(df, entity="entity", time="time", amount="amount",
                                counterparty="merchant", category="category", sequence_length=3)
    r = recommend_feature_profile(p)
    assert p.entity_repeat_row_share == 0.0
    assert p.counterparty_repeat_row_share == 1.0
    assert r.level == "E1"
    assert r.numeric_mode == "continuous"
    assert r.feature_groups == ()


def test_same_timestamp_peers_do_not_count_as_prior_sequence():
    df = pd.DataFrame({
        "entity": ["C1"] * 5,
        "time": [pd.Timestamp("2024-01-01 00:00:00")] * 4 + [pd.Timestamp("2024-01-01 01:00:00")],
        "amount": [1, 2, 3, 4, 5],
    })
    p = profile_feature_support(df, entity="entity", time="time", amount="amount", sequence_length=3)
    # Only the final row has at least 3 strictly earlier observations.
    assert p.entity_sequence_coverage == 0.2


def test_autodata_bridge_and_config_application_do_not_mutate_original():
    from types import SimpleNamespace
    from src.profiling.feature_profile import recommend_for_autodata, apply_feature_recommendation
    df = pd.DataFrame({
        "entity": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "time": pd.date_range("2024-01-01", periods=8, freq="h"),
        "amount": range(8),
        "merchant": ["M1", "M2"] * 4,
    })
    roles = SimpleNamespace(entity="entity", time="time", amount="amount", merchant="merchant", category=None)
    cfg = {"features": {"sequence_length": 3, "enabled_groups": ["temporal"]},
           "representation": {"numeric_mode": "quantile_bin"},
           "adaptive_features": {"min_history_repeat_share": 0.25, "min_sequence_coverage": 0.10,
                                 "min_counterparty_repeat_share": 0.25}}
    rec = recommend_for_autodata(df, roles, cfg)
    changed = apply_feature_recommendation(cfg, rec)
    assert rec.level == "E2"
    assert changed["features"]["enabled_groups"] == list(rec.feature_groups)
    assert cfg["features"]["enabled_groups"] == ["temporal"]
