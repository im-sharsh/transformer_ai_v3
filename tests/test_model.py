import numpy as np
import pandas as pd
import torch

from src.ingestion.roles import detect_roles, schema_for_profiling
from src.ingestion.schema_detector import detect_schema
from src.models.base import basic_data_checks
from src.models.sanity_transformer import SanityTransformerAdapter
from src.preprocessing.levels import DataPreparer, PreparedLevel, infer_roles
from src.representation.tabular_tokenizer import TabularTokenizer
from src.utils.config import load_config

CFG = load_config()


def _prepared_level(df, dataset_id, level, rows=3000, seed=42):
    schema = detect_schema(df, dataset_id)
    roles = detect_roles(df, schema)
    ps = schema_for_profiling(schema, roles, df)
    lr = infer_roles(df, ps, roles)
    dp = DataPreparer(df, ps, lr, CFG, rows=rows, seed=seed)
    dp.prepare_split()
    return dp.build(level)


def test_tabular_tokenizer_ids_in_vocab_and_round_trips(transactions):
    e1 = _prepared_level(transactions, "tx_tok", "E1")
    tok = TabularTokenizer(CFG["representation"]["numeric_bins"], CFG["representation"]["min_category_count"])
    tok.fit(e1.frames["train"], e1.numeric, e1.categorical)
    ids = tok.transform(e1.frames["test"])
    assert ids.shape == (len(e1.frames["test"]), tok.n_positions)
    assert ids.min() >= 0 and ids.max() < tok.vocab_size
    restored = TabularTokenizer.from_dict(tok.to_dict())
    np.testing.assert_array_equal(restored.transform(e1.frames["test"]), ids)


def test_sanity_transformer_trains_and_learns_on_full_demo_data():
    """A larger synthetic set than the shared `transactions` fixture (more cards/days), so the clustered fraud
    bursts reliably land in every split; the shared fixture is small enough that a split can end up with zero
    positives (see test_evaluate_never_fabricates_metrics_on_a_single_class_split, which relies on exactly that)."""
    from src.ingestion.demo_data import make_transactions
    big = make_transactions(n_cards=150, days=120, seed=0)
    e2 = _prepared_level(big, "tx_full", "E2", rows="full")
    adapter = SanityTransformerAdapter(CFG, device="cpu")
    summary = adapter.train(e2, {"epochs": 6, "batch_size": 256, "patience": 6})
    losses = [h["train_loss"] for h in summary["history"]]
    assert losses[-1] < losses[0]
    assert summary["device"] == "cpu"

    checks = adapter.sanity_checks(e2)
    assert all(c["status"] != "fail" for c in checks), checks

    results, pv, pt = adapter.evaluate(e2)
    assert 0.0 <= results["threshold_from_validation"] <= 1.0
    assert results["test"]["pr_auc"] is not None and results["test"]["pr_auc"] > 0.3, \
        "the synthetic fraud burst pattern is designed to be learnable"
    assert len(pv) == len(e2.frames["validation"]) and len(pt) == len(e2.frames["test"])


def test_class_weighting_default_uses_sample_weight_and_can_be_disabled():
    """Audit finding 4: _weight (inverse-probability correction for oversampling) was previously computed but
    never reached the training loss. class_weighting='sample_weight' is now the default; 'none' must
    reproduce the exact old unweighted behaviour so the two are directly comparable."""
    from src.ingestion.demo_data import make_transactions
    big = make_transactions(n_cards=150, days=120, seed=0)
    e1 = _prepared_level(big, "tx_weight", "E1", rows=2000)
    assert e1.frames["train"]["_weight"].nunique() > 1, "fixture must actually have non-uniform sample weights"

    torch.manual_seed(0); np.random.seed(0)
    weighted = SanityTransformerAdapter(CFG, device="cpu")
    weighted.train(e1, {"epochs": 1, "batch_size": 256, "class_weighting": "sample_weight", "seed": 42})

    torch.manual_seed(0); np.random.seed(0)
    unweighted = SanityTransformerAdapter(CFG, device="cpu")
    unweighted.train(e1, {"epochs": 1, "batch_size": 256, "class_weighting": "none", "seed": 42})

    p_weighted = weighted.predict(e1.frames["test"])
    p_unweighted = unweighted.predict(e1.frames["test"])
    assert not np.allclose(p_weighted, p_unweighted), \
        "sample weighting must actually change training, not silently no-op"


def test_class_weighting_rejects_unknown_value():
    from src.ingestion.demo_data import make_transactions
    big = make_transactions(n_cards=80, days=60, seed=0)
    e1 = _prepared_level(big, "tx_weight_bad", "E1", rows=1000)
    adapter = SanityTransformerAdapter(CFG, device="cpu")
    try:
        adapter.train(e1, {"epochs": 1, "class_weighting": "bogus"})
        raise AssertionError("must reject an unknown class_weighting value")
    except ValueError as e:
        assert "class_weighting" in str(e)


def test_sanity_checks_fail_on_empty_data():
    empty = PreparedLevel("E1", {"train": pd.DataFrame({"_target": pd.Series(dtype="float64")}),
                                 "validation": pd.DataFrame({"_target": pd.Series(dtype="float64")}),
                                 "test": pd.DataFrame({"_target": pd.Series(dtype="float64")})}, [], [], {})
    checks = basic_data_checks(empty)
    assert any(c["status"] == "fail" for c in checks)


def test_save_and_load_round_trip_gives_identical_predictions(tmp_path, transactions):
    e1 = _prepared_level(transactions, "tx_save", "E1")
    adapter = SanityTransformerAdapter(CFG, device="cpu")
    adapter.train(e1, {"epochs": 2, "batch_size": 256})
    before = adapter.predict(e1.frames["test"])
    path = adapter.save(tmp_path / "model")

    loaded = SanityTransformerAdapter(CFG, device="cpu").load(path)
    after = loaded.predict(e1.frames["test"])
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_evaluate_never_fabricates_metrics_on_a_single_class_split(transactions):
    """A small subset can end up with zero positives in a split (the synthetic fraud is a rare, clustered
    burst); classification_metrics must report None for AUC there, never a fabricated number."""
    e1 = _prepared_level(transactions, "tx_small", "E1", rows=500)
    adapter = SanityTransformerAdapter(CFG, device="cpu")
    adapter.train(e1, {"epochs": 1, "batch_size": 256})
    results, _, _ = adapter.evaluate(e1)
    for split in ["validation", "test"]:
        m = results[split]
        if m["positives"] == 0:
            assert m["pr_auc"] is None and m["roc_auc"] is None
