import pandas as pd

from src.ingestion.canonical_schema import build_canonical_mapping
from src.ingestion.roles import detect_roles, schema_for_profiling
from src.ingestion.schema_detector import detect_schema
from src.preprocessing.levels import infer_roles


def _generic_frame():
    return pd.DataFrame({
        "sender_account": ["a", "a", "b", "c"],
        "beneficiary": ["m1", "m2", "m1", "m3"],
        "event_time": pd.date_range("2026-01-01", periods=4, freq="h"),
        "transaction_amount": [10.0, 12.0, 800.0, 9.0],
        "payment_type": ["card", "card", "wire", "card"],
        "fraud_flag": [0, 0, 1, 0],
    })


def test_canonical_mapping_with_overrides():
    df = _generic_frame()
    schema = detect_schema(df, "generic")
    dr = detect_roles(df, schema, target="fraud_flag")
    dr = dr.with_overrides(df, entity="sender_account", datetime="event_time")
    ps = schema_for_profiling(schema, dr, df)
    lr = infer_roles(df, ps, dr, hints={
        "amount_column": "transaction_amount",
        "category_column": "payment_type",
        "merchant_column": "beneficiary",
    })
    m = build_canonical_mapping(df, ps, dr, lr)
    assert m.column("target") == "fraud_flag"
    assert m.column("entity") == "sender_account"
    assert m.column("time") == "event_time"
    assert m.column("amount") == "transaction_amount"
    assert m.column("category") == "payment_type"
    assert m.column("counterparty") == "beneficiary"
    assert m.ready(("target", "entity", "time", "amount"))


def test_manual_override_wins_and_is_high_confidence():
    df = _generic_frame()
    schema = detect_schema(df, "generic")
    dr = detect_roles(df, schema, target="fraud_flag")
    m = build_canonical_mapping(df, schema, dr, overrides={"entity": "sender_account", "time": "event_time"})
    assert m.column("entity") == "sender_account"
    assert m.confidence("entity") == 1.0
    assert m.fields["entity"].source == "manual_override"
