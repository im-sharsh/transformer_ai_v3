"""Phase 7: Hugging Face / Nemotron / Custom-API backends behind the same ModelAdapter interface.

Hugging Face tests need `transformers`/`peft` (optional dependencies, see requirements.txt) and network access
to the Hugging Face Hub for a tiny public model; they're skipped, not failed, if either is unavailable.
"""
import http.server
import json
import threading

import numpy as np
import pytest

from src.ingestion.roles import detect_roles, schema_for_profiling
from src.ingestion.schema_detector import detect_schema
from src.models.api_adapter import APIAdapter
from src.models.base import ModelAdapter
from src.models.registry import ADAPTERS, available_backends, get_adapter
from src.models.sanity_transformer import SanityTransformerAdapter
from src.preprocessing.levels import DataPreparer, infer_roles
from src.utils.config import load_config

CFG = load_config()


def _preparer(df, dataset_id, rows=500, seed=42):
    schema = detect_schema(df, dataset_id)
    roles = detect_roles(df, schema)
    ps = schema_for_profiling(schema, roles, df)
    lr = infer_roles(df, ps, roles)
    dp = DataPreparer(df, ps, lr, CFG, rows=rows, seed=seed)
    dp.prepare_split()
    return dp


def test_registry_lists_every_backend_and_all_are_adapters():
    assert set(ADAPTERS) == {"sanity_transformer", "huggingface", "nemotron", "api"}
    for cls in ADAPTERS.values():
        assert issubclass(cls, ModelAdapter)


def test_available_backends_checks_are_real_not_a_static_list():
    hw = {"cuda": False, "gpu_name": None, "gpu_memory_gb": None}
    out = available_backends(hw, CFG)
    by_key = {b["key"]: b for b in out}
    assert by_key["sanity_transformer"]["available"] is True
    # api: unavailable by default (no endpoint configured in config.yaml)
    assert by_key["api"]["available"] is False and "endpoint" in by_key["api"]["message"]
    # nemotron: never available without a CUDA GPU, matching section 13/14 (no fabricated checkpoint support)
    assert by_key["nemotron"]["available"] is False and "GPU" in by_key["nemotron"]["message"]


def test_get_adapter_returns_the_right_class_and_rejects_unknown_keys():
    assert isinstance(get_adapter("sanity_transformer", CFG), SanityTransformerAdapter)
    with pytest.raises(KeyError):
        get_adapter("not_a_real_backend", CFG)


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    """A real local HTTP endpoint matching the API adapter's documented contract, for a genuine round trip."""

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        probs = [0.1 + 0.01 * (i % 5) for i in range(len(body["prompts"]))]
        payload = json.dumps({"probabilities": probs}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def local_api_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _EchoHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_api_adapter_end_to_end_against_a_real_local_endpoint(local_api_server, transactions):
    cfg = load_config()
    cfg["models"]["api"]["endpoint"] = local_api_server
    assert APIAdapter.availability({}, cfg) == (True, f"endpoint {local_api_server}")

    dp = _preparer(transactions, "tx_api", rows=300)
    e1 = dp.build("E1")
    adapter = APIAdapter(cfg)
    summary = adapter.train(e1, {})
    assert summary["train_seconds"] == 0.0                       # honestly reports: inference only, no local training

    checks = adapter.sanity_checks(e1)
    assert all(c["status"] != "fail" for c in checks)

    preds = adapter.predict(e1.frames["test"])
    assert len(preds) == len(e1.frames["test"])
    assert np.isfinite(preds).all() and preds.min() >= 0 and preds.max() <= 1

    results, pv, pt = adapter.evaluate(e1)                       # shared ModelAdapter.evaluate(), unchanged
    assert 0.0 <= results["threshold_from_validation"] <= 1.0


def test_api_adapter_rejects_probabilities_outside_zero_one(transactions):
    class _BadHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            payload = json.dumps({"probabilities": [1.5] * len(body["prompts"])}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _BadHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = load_config()
        cfg["models"]["api"]["endpoint"] = f"http://127.0.0.1:{port}"
        dp = _preparer(transactions, "tx_api_bad", rows=200)
        e1 = dp.build("E1")
        adapter = APIAdapter(cfg)
        adapter.train(e1, {})
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            adapter.predict(e1.frames["test"])
    finally:
        server.shutdown()


hf = pytest.importorskip("transformers", reason="optional dependency, see requirements.txt")
pytest.importorskip("peft", reason="optional dependency, see requirements.txt")


def _hf_reachable() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hf_reachable(), reason="no network access to the Hugging Face Hub")
def test_huggingface_adapter_trains_predicts_and_round_trips_with_a_real_tiny_model(transactions):
    """distilgpt2: a real, small, public GPT-2-family model (not a fabricated checkpoint), small enough to
    LoRA-fine-tune on CPU in a few seconds for this test. GPT-2's LoRA target modules (c_attn) differ from the
    LLaMA-family default (q_proj/...), which is exactly why target_modules is configurable per model."""
    import tempfile
    from pathlib import Path

    from src.models.hf_adapter import HuggingFaceAdapter

    cfg = load_config()
    hcfg = cfg["models"]["huggingface"]
    hcfg.update(model_id="distilgpt2", epochs=1, batch_size=4, gradient_accumulation_steps=1,
               eval_every_steps=10, checkpoint_every_steps=10)

    dp = _preparer(transactions, "tx_hf", rows=150)
    e1 = dp.build("E1")
    assert HuggingFaceAdapter.availability({"cuda": False}, cfg)[0] is True

    with tempfile.TemporaryDirectory() as tmp:
        adapter = HuggingFaceAdapter(cfg)
        checks = adapter.sanity_checks(e1)
        assert all(c["status"] != "fail" for c in checks), checks

        summary = adapter.train(e1, {"run_dir": Path(tmp) / "run"})
        assert summary["history"], "at least one evaluation/checkpoint step should have run"

        preds = adapter.predict(e1.frames["test"])
        assert len(preds) == len(e1.frames["test"]) and np.isfinite(preds).all()

        results, _, _ = adapter.evaluate(e1)
        assert 0.0 <= results["threshold_from_validation"] <= 1.0

        path = adapter.save(Path(tmp) / "saved")
        loaded = HuggingFaceAdapter(cfg).load(path)
        np.testing.assert_allclose(preds, loaded.predict(e1.frames["test"]), atol=1e-4)
