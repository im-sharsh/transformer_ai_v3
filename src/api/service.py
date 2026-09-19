"""Pure service functions used by the HTTP API and future SaaS orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ingestion.canonical_schema import build_canonical_mapping
from src.ingestion.loader import load_dataset
from src.ingestion.roles import detect_roles, schema_for_profiling
from src.ingestion.schema_detector import detect_schema
from src.preprocessing.levels import infer_roles
from src.profiling.feature_profile import recommend_for_autodata
from src.quality.quality_engine import assess_quality
from src.utils.config import load_config


def analyze_dataset(path: str | Path, *, target: str | None = None, overrides: dict[str, str] | None = None,
                    config: dict | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    ds = load_dataset(path, save_metadata=False)
    schema = detect_schema(ds.df, ds.metadata.dataset_id, sample_size=cfg["schema"]["sample_size"])
    roles = detect_roles(ds.df, schema, target=target)
    overrides = dict(overrides or {})
    roles = roles.with_overrides(
        ds.df,
        target=overrides.get("target", target) if (overrides.get("target") or target) else None,
        entity=overrides.get("entity"),
        datetime=overrides.get("time"),
    )
    ps = schema_for_profiling(schema, roles, ds.df)
    hints = {
        "amount_column": overrides.get("amount"),
        "category_column": overrides.get("category"),
        "merchant_column": overrides.get("counterparty"),
    }
    level_roles = infer_roles(ds.df, ps, roles, hints=hints) if roles.target else None
    mapping = build_canonical_mapping(ds.df, ps, roles, level_roles, overrides=overrides)
    recommendation = recommend_for_autodata(ds.df, level_roles, cfg).to_dict() if level_roles else None
    return {
        "metadata": ds.metadata.to_dict() if hasattr(ds.metadata, "to_dict") else ds.metadata.__dict__,
        "schema": ps.to_dict(),
        "canonical_mapping": mapping.to_dict(),
        "adaptive_recommendation": recommendation,
    }


def validate_dataset(path: str | Path, *, target: str | None = None, overrides: dict[str, str] | None = None,
                     config: dict | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    ds = load_dataset(path, save_metadata=False)
    schema = detect_schema(ds.df, ds.metadata.dataset_id, sample_size=cfg["schema"]["sample_size"])
    roles = detect_roles(ds.df, schema, target=target)
    overrides = dict(overrides or {})
    roles = roles.with_overrides(ds.df, target=overrides.get("target", target) if (overrides.get("target") or target) else None,
                                entity=overrides.get("entity"), datetime=overrides.get("time"))
    ps = schema_for_profiling(schema, roles, ds.df)
    return assess_quality(ds.df, ps).to_dict()


def prepare_dataset_archive(path: str | Path, *, target: str | None = None, overrides: dict[str, str] | None = None,
                            level: str = "auto", rows: int | str = 200_000,
                            config: dict | None = None) -> tuple[bytes, dict[str, Any]]:
    """Prepare E1/E2 data and return a ZIP archive plus manifest.

    This is a synchronous v1 endpoint intended for pilot-sized jobs. Large production jobs
    should move to an asynchronous job/queue API rather than keeping an HTTP request open.
    """
    import io
    import json
    import zipfile

    from src.preprocessing.levels import DataPreparer
    from src.profiling.feature_profile import apply_feature_recommendation

    cfg = config or load_config()
    overrides = dict(overrides or {})
    ds = load_dataset(path, save_metadata=False)
    schema = detect_schema(ds.df, ds.metadata.dataset_id, sample_size=cfg["schema"]["sample_size"])
    roles = detect_roles(ds.df, schema, target=target)
    roles = roles.with_overrides(
        ds.df,
        target=overrides.get("target", target) if (overrides.get("target") or target) else None,
        entity=overrides.get("entity"),
        datetime=overrides.get("time"),
    )
    if not roles.target:
        raise ValueError("/prepare currently requires a supervised target; supply target or a target override")
    ps = schema_for_profiling(schema, roles, ds.df)
    hints = {
        "amount_column": overrides.get("amount"),
        "category_column": overrides.get("category"),
        "merchant_column": overrides.get("counterparty"),
    }
    level_roles = infer_roles(ds.df, ps, roles, hints=hints)
    mapping = build_canonical_mapping(ds.df, ps, roles, level_roles, overrides=overrides)
    recommendation = recommend_for_autodata(ds.df, level_roles, cfg)

    chosen = level.upper()
    if chosen == "AUTO":
        chosen = recommendation.level
    if chosen not in {"E1", "E2"}:
        raise ValueError("level must be auto, E1 or E2")

    run_cfg = apply_feature_recommendation(cfg, recommendation) if level.lower() == "auto" else cfg
    if chosen == "E1":
        # Conservative E1 contract validated in our benchmarks.
        import copy
        run_cfg = copy.deepcopy(run_cfg)
        run_cfg.setdefault("representation", {})["numeric_mode"] = "continuous"

    preparer = DataPreparer(ds.df, ps, level_roles, run_cfg, rows=rows, seed=run_cfg["project"]["seed"])
    preparer.prepare_split()
    prepared = preparer.build(chosen)

    manifest = {
        "dataset_id": ds.metadata.dataset_id,
        "source_file": ds.metadata.file_name,
        "level": chosen,
        "numeric_mode": run_cfg.get("representation", {}).get("numeric_mode"),
        "feature_groups": run_cfg.get("features", {}).get("enabled_groups", []) if chosen == "E2" else [],
        "canonical_mapping": mapping.to_dict(),
        "adaptive_recommendation": recommendation.to_dict(),
        "split": preparer.split_info,
        "features": {"numeric": prepared.numeric, "categorical": prepared.categorical},
        "point_in_time": prepared.info.get("point_in_time"),
    }

    # Prefer Parquet for typed/model-ready interchange.  Fall back to CSV when the
    # optional parquet engine is unavailable; the manifest records the exact format.
    try:
        import pyarrow  # noqa: F401
        archive_format = "parquet"
    except ImportError:
        archive_format = "csv"
    manifest["archive_format"] = archive_format

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
        for split, frame in prepared.frames.items():
            if archive_format == "parquet":
                payload = io.BytesIO()
                frame.to_parquet(payload, index=False)
                zf.writestr(f"{split}.parquet", payload.getvalue())
            else:
                zf.writestr(f"{split}.csv", frame.to_csv(index=False).encode("utf-8"))
    return buffer.getvalue(), manifest
