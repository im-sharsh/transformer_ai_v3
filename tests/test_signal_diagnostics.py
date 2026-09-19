from src.evaluation.signal_diagnostics import compare_levels, diagnose_level


def _prepared(transactions):
    import yaml
    from src.ingestion.roles import detect_roles, schema_for_profiling
    from src.ingestion.schema_detector import detect_schema
    from src.preprocessing.levels import DataPreparer, infer_roles

    cfg = yaml.safe_load(open("config.yaml"))
    schema = detect_schema(transactions, "diag")
    roles = detect_roles(transactions, schema, target="is_fraud")
    ps = schema_for_profiling(schema, roles, transactions)
    inferred = infer_roles(transactions, ps, roles)
    prep = DataPreparer(transactions, ps, inferred, cfg, rows=1500, seed=42)
    prep.prepare_split()
    return cfg, prep


def test_signal_diagnostics_are_train_fitted_and_report_all_levels(transactions):
    cfg, prep = _prepared(transactions)
    levels = {x: prep.build(x) for x in ["E0", "E1", "E2"]}
    table = compare_levels(levels, cfg["representation"])
    assert set(table["level"]) == {"E0", "E1", "E2"}
    assert (table["n_features"] > 0).all()
    assert table["best_univariate_test_auc"].between(0.5, 1.0).all()


def test_categorical_unknown_rate_is_measured(transactions):
    cfg, prep = _prepared(transactions)
    d = diagnose_level(prep.build("E0"), cfg["representation"])
    cats = [f for f in d["features"] if f["kind"] == "categorical"]
    assert cats
    assert all(f["test_unknown_rate"] is not None for f in cats)
