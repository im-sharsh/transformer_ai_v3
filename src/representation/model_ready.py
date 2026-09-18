"""Build inspectable/downloadable artifacts that mirror the built-in Transformer's inputs.

This module deliberately sits between preprocessing and the model.  It fits representation state on the
training split only, transforms train/validation/test with that state, and packages both the actual arrays and
the metadata needed to explain/reproduce them.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass

import pandas as pd

from src.preprocessing.levels import META, PreparedLevel
from src.representation.tabular_tokenizer import TabularTokenizer


@dataclass
class ModelReadyArtifacts:
    level: str
    frames: dict[str, pd.DataFrame]
    schema: dict
    feature_dictionary: dict
    preprocessing_config: dict
    representation_config: dict
    processing_report: dict
    tokenizer: TabularTokenizer


def _tokenizer_from_config(config: dict, prepared: PreparedLevel) -> TabularTokenizer:
    rcfg = config["representation"]
    return TabularTokenizer(
        rcfg["numeric_bins"],
        rcfg["min_category_count"],
        numeric_mode=rcfg.get("numeric_mode", "quantile_bin"),
        coarse_bins=rcfg.get("numeric_coarse_bins", 0),
        numeric_clip=rcfg.get("numeric_clip"),
        continuous_features=rcfg.get("continuous_features"),
    ).fit(prepared.frames["train"], prepared.numeric, prepared.categorical)


def _position_dictionary(tokenizer: TabularTokenizer) -> list[dict]:
    positions = [{"position": 0, "name": "CLS", "source_feature": None, "feature_type": "special",
                  "representation": "CLS token"}]
    continuous = set(tokenizer.numeric_positions)
    for pos, feature in enumerate(tokenizer.numeric, start=1):
        positions.append({
            "position": pos,
            "name": f"token_{pos:03d}",
            "source_feature": feature,
            "feature_type": "numeric",
            "representation": "continuous standardized value + token marker" if pos in continuous
                              else "train-fitted quantile-bin token",
        })
    start = 1 + len(tokenizer.numeric)
    for offset, feature in enumerate(tokenizer.categorical):
        pos = start + offset
        positions.append({
            "position": pos,
            "name": f"token_{pos:03d}",
            "source_feature": feature,
            "feature_type": "categorical",
            "representation": "train-fitted categorical token; unseen values use unknown token",
        })
    return positions


def build_model_ready(prepared: PreparedLevel, config: dict) -> ModelReadyArtifacts:
    """Fit the representation on train only and materialize exactly what the built-in model consumes."""
    tokenizer = _tokenizer_from_config(config, prepared)
    positions = _position_dictionary(tokenizer)
    token_cols = [f"token_{i:03d}" for i in range(tokenizer.n_positions)]
    continuous = bool(tokenizer.numeric_positions)
    frames: dict[str, pd.DataFrame] = {}

    for split, source in prepared.frames.items():
        ids = tokenizer.transform(source)
        out = source[[c for c in META if c in source]].copy().reset_index(drop=True)
        # A split is already encoded by the filename and keeping it in every row is redundant.
        out = out.drop(columns=["_split"], errors="ignore")
        token_df = pd.DataFrame(ids, columns=token_cols)
        out = pd.concat([out, token_df], axis=1)
        if continuous:
            values, mask = tokenizer.transform_numeric(source)
            value_df = pd.DataFrame(values, columns=[f"numeric_value_{i:03d}" for i in range(tokenizer.n_positions)])
            mask_df = pd.DataFrame(mask, columns=[f"numeric_mask_{i:03d}" for i in range(tokenizer.n_positions)])
            out = pd.concat([out, value_df, mask_df], axis=1)
        frames[split] = out

    schema = {
        "level": prepared.level,
        "splits": {name: {"rows": int(len(frame)),
                            "columns": [{"name": str(c), "dtype": str(frame[c].dtype)} for c in frame.columns]}
                   for name, frame in frames.items()},
        "target_column": "_target",
        "weight_column": "_weight",
        "token_columns": token_cols,
        "numeric_value_columns": ([f"numeric_value_{i:03d}" for i in range(tokenizer.n_positions)] if continuous else []),
        "numeric_mask_columns": ([f"numeric_mask_{i:03d}" for i in range(tokenizer.n_positions)] if continuous else []),
    }
    feature_dictionary = {
        "sequence_positions": positions,
        "numeric_features": list(prepared.numeric),
        "categorical_features": list(prepared.categorical),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": tokenizer.to_dict(),
    }
    preprocessing_config = {
        "processing": config.get("processing", {}),
        "features": config.get("features", {}),
        "split": config.get("split", {}),
        "sampling": config.get("sampling", {}),
        "level_steps": prepared.info.get("steps", []),
    }
    representation_config = {
        **config.get("representation", {}),
        "n_positions": tokenizer.n_positions,
        "continuous_numeric_positions": tokenizer.numeric_positions,
        "fit_split": "train",
        "transform_splits": ["train", "validation", "test"],
    }
    processing_report = {
        "level": prepared.level,
        "name": prepared.info.get("name"),
        "features": len(prepared.features),
        "rows": prepared.info.get("rows", {}),
        "positives": prepared.info.get("positives", {}),
        "split": prepared.info.get("split", {}),
        "point_in_time": prepared.info.get("point_in_time"),
        "leakage": prepared.info.get("leakage"),
    }
    return ModelReadyArtifacts(prepared.level, frames, schema, feature_dictionary, preprocessing_config,
                               representation_config, processing_report, tokenizer)



def serialize_model_ready_frame(frame: pd.DataFrame, split: str) -> tuple[str, bytes, str]:
    """Prefer parquet (the project dependency), but degrade to CSV if the optional engine is unavailable.

    This makes the product usable in partial environments while keeping parquet as the normal installed-project
    artifact format. The chosen format is explicit in the returned filename; nothing is silently mislabeled.
    """
    try:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        return f"{split}.parquet", buffer.getvalue(), "application/octet-stream"
    except (ImportError, ModuleNotFoundError):
        payload = frame.to_csv(index=False).encode("utf-8")
        return f"{split}.csv", payload, "text/csv"

def model_ready_zip_bytes(artifacts: ModelReadyArtifacts) -> bytes:
    """Create the complete Step-11 package without writing temporary files to the project tree."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        data_files = {}
        for split, frame in artifacts.frames.items():
            filename, payload, _ = serialize_model_ready_frame(frame, split)
            data_files[split] = filename
            zf.writestr(filename, payload)
        artifacts.processing_report["model_ready_files"] = data_files
        metadata = {
            "schema.json": artifacts.schema,
            "feature_dictionary.json": artifacts.feature_dictionary,
            "preprocessing_config.json": artifacts.preprocessing_config,
            "representation_config.json": artifacts.representation_config,
            "processing_report.json": artifacts.processing_report,
        }
        for name, payload in metadata.items():
            zf.writestr(name, json.dumps(payload, indent=2, default=str))
    return buffer.getvalue()
