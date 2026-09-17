from src.evaluation.experiment import ExperimentResult, comparison_table, list_experiments, run_comparison, save_experiment
from src.ingestion.demo_data import make_transactions
from src.ingestion.roles import detect_roles, schema_for_profiling
from src.ingestion.schema_detector import detect_schema
from src.preprocessing.levels import DataPreparer, infer_roles
from src.utils.config import load_config

CFG = load_config()
FAST = {"epochs": 2, "patience": 5}


def _preparer(df, dataset_id, rows=5000, seed=42):
    schema = detect_schema(df, dataset_id)
    roles = detect_roles(df, schema)
    ps = schema_for_profiling(schema, roles, df)
    lr = infer_roles(df, ps, roles)
    dp = DataPreparer(df, ps, lr, CFG, rows=rows, seed=seed)
    dp.prepare_split()
    return dp


def test_run_comparison_uses_the_same_split_and_seed_for_every_level():
    big = make_transactions(n_cards=150, days=120, seed=0)
    dp = _preparer(big, "tx_exp", rows="full")
    results = run_comparison(dp, ["E0", "E1"], CFG, settings=FAST, device="cpu")
    assert len(results) == 2
    assert {r.seed for r in results} == {CFG["models"]["sanity_transformer"]["seed"]}
    for r in results:
        assert r.rows["train"] == dp.split_info["sample"]["train"]["rows"]
        assert r.rows["test"] == dp.split_info["sample"]["test"]["rows"]


def test_comparison_table_has_one_row_per_level_in_e0_e1_e2_order():
    big = make_transactions(n_cards=150, days=120, seed=0)
    dp = _preparer(big, "tx_exp2", rows=3000)
    results = run_comparison(dp, ["E2", "E0", "E1"], CFG, settings=FAST, device="cpu")   # deliberately out of order
    table = comparison_table(results)
    assert list(table["Level"]) == ["E0", "E1", "E2"]
    assert {"Test PR-AUC", "Test ROC-AUC", "Test Precision", "Test Recall", "Test F1"} <= set(table.columns)


def test_save_and_list_experiments_round_trip(tmp_path):
    big = make_transactions(n_cards=80, days=60, seed=1)
    dp = _preparer(big, "tx_exp3", rows=2000)
    results = run_comparison(dp, ["E0"], CFG, settings=FAST, device="cpu")
    path = save_experiment(results, tmp_path, "tx_exp3", dp.split_info)
    assert path.exists()
    loaded = list_experiments(tmp_path)
    assert len(loaded) == 1 and loaded[0]["dataset_id"] == "tx_exp3"
    restored = ExperimentResult.from_dict(loaded[0]["results"][0])
    assert restored.level == "E0" and restored.metrics == results[0].metrics


def test_list_experiments_is_empty_and_not_fabricated_when_none_saved(tmp_path):
    assert list_experiments(tmp_path) == []


def test_experiments_are_never_overwritten(tmp_path):
    big = make_transactions(n_cards=80, days=60, seed=1)
    dp = _preparer(big, "tx_exp4", rows=1000)
    results = run_comparison(dp, ["E0"], CFG, settings=FAST, device="cpu")
    p1 = save_experiment(results, tmp_path, "tx_exp4", dp.split_info)
    p2 = save_experiment(results, tmp_path, "tx_exp4", dp.split_info)
    assert p1 != p2
    assert len(list_experiments(tmp_path)) == 2
