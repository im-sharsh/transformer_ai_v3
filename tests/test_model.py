import numpy as np
import pandas as pd
import torch
from pathlib import Path

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


def test_continuous_numeric_mode_preserves_magnitude_that_quantile_bin_discards(transactions):
    """Audit R2: two different raw amounts landing in the same quantile bin are indistinguishable in
    'quantile_bin' mode (proven in the audit) but must remain distinguishable in 'continuous' mode."""
    e1 = _prepared_level(transactions, "tx_cont", "E1")
    tok = TabularTokenizer(8, CFG["representation"]["min_category_count"], numeric_mode="continuous")
    tok.fit(e1.frames["train"], e1.numeric, e1.categorical)
    assert tok.edges == {}, "pure continuous mode (coarse_bins=0) must not build any bin edges"
    ids = tok.transform(e1.frames["test"])
    values, mask = tok.transform_numeric(e1.frames["test"])
    assert values.shape == ids.shape == mask.shape
    num_pos = tok.numeric_positions
    assert num_pos and num_pos == list(range(1, 1 + len(tok.numeric)))
    # Every numeric position's token id is either the per-feature missing marker or PAD -- never a value-coded id.
    for j in num_pos:
        c = tok.numeric[j - 1]
        allowed = {tok.special[c]["missing"], 0}
        assert set(np.unique(ids[:, j]).tolist()) <= allowed, (c, np.unique(ids[:, j]))
    present = mask[:, num_pos[0]] > 0
    assert present.any() and len(np.unique(values[present, num_pos[0]])) > 8, \
        "continuous mode must keep more distinct values than the 8 quantile bins it was deliberately given"

    restored = TabularTokenizer.from_dict(tok.to_dict())
    rv, rm = restored.transform_numeric(e1.frames["test"])
    np.testing.assert_array_equal(restored.transform(e1.frames["test"]), ids)
    np.testing.assert_allclose(rv, values); np.testing.assert_array_equal(rm, mask)


def test_continuous_mode_with_coarse_bins_keeps_both_signals():
    """Audit R3: continuous value + a coarse bin token at the same position, not a new position."""
    transactions_small = pd.DataFrame({"amt": np.linspace(1, 1000, 200), "cat": ["a", "b"] * 100})
    tok = TabularTokenizer(16, 1, numeric_mode="continuous", coarse_bins=4)
    tok.fit(transactions_small, ["amt"], ["cat"])
    assert "amt" in tok.edges and len(tok.vocab["amt"]) == 4, "coarse_bins=4 must build exactly 4 bin ids"
    ids = tok.transform(transactions_small)
    values, mask = tok.transform_numeric(transactions_small)
    assert len(np.unique(ids[:, 1])) == 4                       # 4 distinct coarse bin tokens used
    assert len(np.unique(values[:, 1])) > 4                     # but the continuous value still varies within each bin


def test_continuous_mode_clips_extreme_standardized_values():
    """Diagnosed cause of R2's E2 regression (audit session): a handful of engineered ratio/z-score features
    have standardized values up to |z|~30; numeric_clip caps them before they reach the model."""
    df = pd.DataFrame({"amt": np.concatenate([np.random.default_rng(0).normal(50, 10, 199), [100000.0]]),  # one wild outlier
                       "cat": ["a", "b"] * 100})
    tok = TabularTokenizer(16, 1, numeric_mode="continuous", numeric_clip=5.0)
    tok.fit(df, ["amt"], ["cat"])
    values, mask = tok.transform_numeric(df)
    assert values[:, 1].max() == 5.0 and values[:, 1].min() >= -5.0
    unclipped = TabularTokenizer(16, 1, numeric_mode="continuous", numeric_clip=None)
    unclipped.fit(df, ["amt"], ["cat"])
    uv, _ = unclipped.transform_numeric(df)
    assert uv[:, 1].max() > 100, "the unclipped tokenizer must still show the raw extreme value for comparison"
    restored = TabularTokenizer.from_dict(tok.to_dict())
    assert restored.numeric_clip == 5.0
    rv, _ = restored.transform_numeric(df)
    np.testing.assert_allclose(rv, values)


def test_sanity_transformer_learns_with_continuous_numeric_mode():
    """The full train/predict/save/load path must work end to end with numeric_mode='continuous', not just
    the tokenizer in isolation."""
    from src.ingestion.demo_data import make_transactions
    big = make_transactions(n_cards=150, days=120, seed=0)
    e2 = _prepared_level(big, "tx_continuous_e2", "E2", rows="full")
    cfg = {**CFG, "representation": {**CFG["representation"], "numeric_mode": "continuous"}}
    adapter = SanityTransformerAdapter(cfg, device="cpu")
    summary = adapter.train(e2, {"epochs": 4, "batch_size": 256, "patience": 4})
    assert adapter.model.has_continuous_numeric
    assert summary["history"][-1]["train_loss"] < summary["history"][0]["train_loss"]

    checks = adapter.sanity_checks(e2)
    assert all(c["status"] != "fail" for c in checks), checks

    results, pv, pt = adapter.evaluate(e2)
    assert results["test"]["pr_auc"] is not None and results["test"]["pr_auc"] > 0.2

    import tempfile
    path = adapter.save(Path(tempfile.mkdtemp()) / "model")
    loaded = SanityTransformerAdapter(cfg, device="cpu").load(path)
    np.testing.assert_allclose(adapter.predict(e2.frames["test"]), loaded.predict(e2.frames["test"]), atol=1e-6)


def test_continuous_mode_forward_requires_numeric_tensors():
    """A model built with numeric_positions must refuse a bare forward() call rather than silently ignore
    the numeric channel."""
    from src.models.sanity_transformer import SanityTransformerModel
    m = SanityTransformerModel(vocab_size=10, n_positions=3, d_model=8, n_heads=2, n_layers=1,
                               dim_feedforward=16, numeric_positions=[1])
    ids = torch.zeros((2, 3), dtype=torch.long)
    try:
        m(ids)
        raise AssertionError("must require numeric_values/numeric_mask when numeric_positions is set")
    except ValueError as e:
        assert "numeric_values" in str(e)


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


def test_class_weighting_option_actually_changes_training_and_can_be_selected():
    """Audit finding 4: _weight (inverse-probability correction for oversampling) was previously computed but
    never reached the training loss. class_weighting='sample_weight' was implemented as the fix and initially
    made the default — then measured (3 seeds, synthetic data, outside this test) to cause a large, consistent
    PR-AUC regression at severe class imbalance, so 'none' (the original behaviour) was restored as the
    default. Both options remain implemented and selectable for direct comparison, which is what this test
    guards: 'sample_weight' must actually change training, not silently no-op, whichever one is the default."""
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
