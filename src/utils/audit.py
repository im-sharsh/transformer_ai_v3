"""Transformation audit log, cell-level change log, and dataset versioning."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

from src.ingestion.schema_detector import mask_value


class AuditLog:
    """One entry per pipeline operation (machine-readable, for reproducibility)."""

    def __init__(self):
        self.entries: list[dict] = []

    def add(self, step: str, operation: str, reason: str, input_rows: int, output_rows: int,
            columns_before: int, columns_after: int, records_modified: int = 0,
            records_removed: int = 0, **details):
        self.entries.append({
            "step": step, "operation": operation, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input_rows": int(input_rows), "output_rows": int(output_rows),
            "columns_before": int(columns_before), "columns_after": int(columns_after),
            "records_modified": int(records_modified), "records_removed": int(records_removed),
            "reason": reason, "details": details,
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)

    def save(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        j, c = out_dir / "audit_log.json", out_dir / "audit_log.csv"
        j.write_text(json.dumps(self.entries, indent=2, default=str))
        frame = self.to_frame()
        if len(frame):
            frame.assign(details=frame["details"].map(lambda d: json.dumps(d, default=str))).to_csv(c, index=False)
        return [j, c]


class ChangeLog:
    """Cell-level record of value changes: row, column, original, new, operation, reason.

    Sensitive values are masked. Very large operations are summarised with examples only.
    """

    def __init__(self, max_cells_per_operation: int = 50_000, example_cells: int = 20):
        self.frames: list[pd.DataFrame] = []
        self.summaries: list[dict] = []
        self.max_cells = max_cells_per_operation
        self.examples = example_cells

    def add(self, row_ids: pd.Series, column: str, original: pd.Series, new: pd.Series,
            operation: str, reason: str, pii_type: str | None = None):
        n = len(row_ids)
        if n == 0:
            return
        keep = n if n <= self.max_cells else self.examples
        mask = (lambda v: mask_value(v, pii_type)) if pii_type else (lambda v: None if pd.isna(v) else str(v)[:80])
        frame = pd.DataFrame({
            "row_id": row_ids.iloc[:keep].astype(str).values, "column": column,
            "original_value": original.iloc[:keep].map(mask).values,
            "new_value": new.iloc[:keep].map(mask).values,
            "operation": operation, "reason": reason,
        })
        self.frames.append(frame)
        self.summaries.append({"column": column, "operation": operation, "cells_changed": int(n),
                               "cells_logged": int(keep), "summarised": n > self.max_cells})

    def to_frame(self) -> pd.DataFrame:
        cols = ["row_id", "column", "original_value", "new_value", "operation", "reason"]
        return pd.concat(self.frames, ignore_index=True) if self.frames else pd.DataFrame(columns=cols)

    def save(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        p, s = out_dir / "change_log.parquet", out_dir / "change_log_summary.json"
        self.to_frame().astype(str).to_parquet(p, index=False)
        s.write_text(json.dumps(self.summaries, indent=2))
        return [p, s]


def next_version(processed_dir: str | Path, prefix: str = "dataset_v") -> str:
    """dataset_v001, dataset_v002, ... Existing versions are never overwritten."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    numbers = [int(m.group(1)) for p in processed_dir.iterdir()
               if (m := re.fullmatch(rf"{prefix}(\d+)", p.name))]
    return f"{prefix}{max(numbers, default=0) + 1:03d}"
