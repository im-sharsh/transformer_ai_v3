import pandas as pd

from src.api.service import analyze_dataset


def test_analyze_dataset_returns_canonical_mapping(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({
        "account_id": ["a", "a", "b", "b", "c", "c"],
        "event_time": pd.date_range("2026-01-01", periods=6, freq="h"),
        "amount": [1, 2, 3, 4, 5, 6],
        "type": ["a", "b", "a", "b", "a", "b"],
        "fraud": [0, 0, 0, 1, 0, 0],
    }).to_csv(p, index=False)
    out = analyze_dataset(p, target="fraud", overrides={"entity": "account_id", "time": "event_time", "amount": "amount", "category": "type"})
    assert out["canonical_mapping"]["fields"]["target"]["column"] == "fraud"
    assert out["canonical_mapping"]["fields"]["entity"]["column"] == "account_id"
    assert out["adaptive_recommendation"] is not None


def test_prepare_dataset_archive_contains_manifest_and_splits(tmp_path):
    import io, zipfile, json
    from src.api.service import prepare_dataset_archive
    p = tmp_path / "prep.csv"
    n = 300
    pd.DataFrame({
        "account_id": [f"a{i%20}" for i in range(n)],
        "event_time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "amount": [float((i % 17) + 1) for i in range(n)],
        "type": ["card" if i % 2 else "wire" for i in range(n)],
        "fraud": [1 if i % 47 == 0 else 0 for i in range(n)],
    }).to_csv(p, index=False)
    payload, manifest = prepare_dataset_archive(
        p, target="fraud", overrides={"entity":"account_id", "time":"event_time", "amount":"amount", "category":"type"},
        level="E1", rows=250,
    )
    assert manifest["level"] == "E1"
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names = set(z.namelist())
        assert "manifest.json" in names
        m = json.loads(z.read("manifest.json"))
        ext = m["archive_format"]
        assert {f"train.{ext}", f"validation.{ext}", f"test.{ext}"} <= names
        assert m["canonical_mapping"]["fields"]["entity"]["column"] == "account_id"
