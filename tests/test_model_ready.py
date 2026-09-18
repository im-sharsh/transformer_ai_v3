import io
import json
import zipfile

import pandas as pd

from src.ingestion.roles import detect_roles, schema_for_profiling
from src.ingestion.schema_detector import detect_schema
from src.preprocessing.levels import DataPreparer, infer_roles
from src.representation.model_ready import build_model_ready, model_ready_zip_bytes
from src.utils.config import load_config


def _prepared(transactions, rows=1200, level="E1"):
    cfg = load_config()
    schema = detect_schema(transactions, dataset_id="model-ready-test")
    roles = detect_roles(transactions, schema, target="is_fraud")
    roles = roles.with_overrides(transactions, entity="cc_num", datetime="trans_date_trans_time",
                                 task="binary_classification")
    ps = schema_for_profiling(schema, roles, transactions)
    inferred = infer_roles(transactions, ps, roles)
    preparer = DataPreparer(transactions, ps, inferred, cfg, rows=min(rows, len(transactions)), seed=42)
    preparer.prepare_split()
    return cfg, preparer, preparer.build(level)


def test_model_ready_is_exact_tokenizer_output_and_has_metadata(transactions):
    cfg, _, prepared = _prepared(transactions)
    artifacts = build_model_ready(prepared, cfg)
    train = artifacts.frames["train"]
    expected = artifacts.tokenizer.transform(prepared.frames["train"])
    token_cols = artifacts.schema["token_columns"]
    assert train[token_cols].to_numpy().tolist() == expected.tolist()
    assert {"_row_id", "_target", "_weight", "_new_entity"}.issubset(train.columns)
    assert artifacts.representation_config["fit_split"] == "train"
    assert len(artifacts.feature_dictionary["sequence_positions"]) == artifacts.tokenizer.n_positions


def test_model_ready_zip_contains_step_11_artifacts(transactions):
    cfg, _, prepared = _prepared(transactions)
    artifacts = build_model_ready(prepared, cfg)
    payload = model_ready_zip_bytes(artifacts)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
        assert {"schema.json", "feature_dictionary.json", "preprocessing_config.json",
                "representation_config.json", "processing_report.json"} <= names
        for split in ["train", "validation", "test"]:
            assert f"{split}.parquet" in names or f"{split}.csv" in names
        schema = json.loads(zf.read("schema.json"))
        assert schema["level"] == prepared.level
        if "train.parquet" in names:
            train = pd.read_parquet(io.BytesIO(zf.read("train.parquet")))
        else:
            train = pd.read_csv(io.BytesIO(zf.read("train.csv")))
        assert len(train) == len(prepared.frames["train"])


def test_custom_subset_count_is_exact_and_reused_across_levels(transactions):
    requested = min(777, len(transactions) - 1)
    cfg, preparer, _ = _prepared(transactions, rows=requested, level="E0")
    assert len(preparer.sample_ids) == requested
    levels = [preparer.build(level) for level in ["E0", "E1", "E2"]]
    for split in ["train", "validation", "test"]:
        ids = [set(level.frames[split]["_row_id"]) for level in levels]
        assert ids[0] == ids[1] == ids[2]
    assert sum(len(levels[0].frames[s]) for s in ["train", "validation", "test"]) == requested
