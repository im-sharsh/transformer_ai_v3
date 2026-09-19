import pandas as pd
import pytest

from src.ingestion.adapters import PAYSIM_EVENT_TIME, adapt_paysim, is_paysim_schema


def _frame():
    return pd.DataFrame({
        "step": [1, 1, 3],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [10.0, 200.0, 50.0],
        "nameOrig": ["C1", "C2", "C1"],
        "oldbalanceOrg": [100, 500, 90],
        "newbalanceOrig": [90, 300, 40],
        "nameDest": ["M1", "C9", "M1"],
        "oldbalanceDest": [0, 20, 10],
        "newbalanceDest": [10, 220, 60],
        "isFraud": [0, 1, 0],
        "isFlaggedFraud": [0, 0, 0],
    })


def test_paysim_strict_adapter_maps_time_and_removes_shortcut_fields():
    df = _frame()
    assert is_paysim_schema(df)
    a = adapt_paysim(df, profile="strict", anchor_time="2020-01-01")
    assert a.target == "isFraud"
    assert a.entity == "nameOrig"
    assert a.datetime == PAYSIM_EVENT_TIME
    assert a.hints["amount_column"] == "amount"
    assert a.hints["category_column"] == "type"
    assert a.hints["merchant_column"] == "nameDest"
    assert "isFlaggedFraud" not in a.df
    assert "oldbalanceOrg" not in a.df
    assert a.df[PAYSIM_EVENT_TIME].iloc[0] == pd.Timestamp("2020-01-01 01:00:00")
    assert a.df[PAYSIM_EVENT_TIME].iloc[2] == pd.Timestamp("2020-01-01 03:00:00")


def test_paysim_full_research_keeps_balances_but_not_flag():
    a = adapt_paysim(_frame(), profile="full_research")
    assert "oldbalanceOrg" in a.df
    assert "newbalanceDest" in a.df
    assert "isFlaggedFraud" not in a.df


def test_paysim_bad_profile_and_bad_step_fail_fast():
    with pytest.raises(ValueError):
        adapt_paysim(_frame(), profile="unknown")
    df = _frame()
    df.loc[1, "step"] = "bad"
    with pytest.raises(ValueError, match="must be numeric"):
        adapt_paysim(df)
