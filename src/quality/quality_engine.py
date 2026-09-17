"""Data quality engine: runs checks, scores five dimensions transparently, keeps row-level issues."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.schema_detector import SchemaReport
from src.profiling.profiler import parse_datetime_column
from src.quality import validators as V

logger = logging.getLogger("QUALITY")

DIMENSIONS = ("completeness", "validity", "consistency", "uniqueness", "integrity")
DEFAULT_WEIGHTS = {"completeness": 0.25, "validity": 0.25, "consistency": 0.20,
                   "uniqueness": 0.15, "integrity": 0.15}
GEO_RANGES = {"lat": (-90, 90), "latitude": (-90, 90), "long": (-180, 180), "lon": (-180, 180),
              "lng": (-180, 180), "longitude": (-180, 180)}


@dataclass
class QualityReport:
    dataset_id: str
    assessed_at: str
    seconds: float
    rows: int
    weights: dict
    scores: dict
    checks: list
    records_with_issues: int
    systematic_issues: list
    notes: list = field(default_factory=list)
    row_issues: pd.DataFrame | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {"dataset_id": self.dataset_id, "assessed_at": self.assessed_at, "seconds": self.seconds,
                "rows": self.rows, "weights": self.weights, "scores": self.scores,
                "records_with_issues": self.records_with_issues, "systematic_issues": self.systematic_issues,
                "notes": self.notes, "checks": [c.to_dict() for c in self.checks]}

    def save(self, out_dir: str | Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {"json": out_dir / f"quality_{self.dataset_id}.json", "md": out_dir / f"quality_{self.dataset_id}.md"}
        paths["json"].write_text(json.dumps(self.to_dict(), indent=2, default=str))
        paths["md"].write_text(self.to_markdown())
        if self.row_issues is not None and len(self.row_issues):
            paths["row_issues"] = out_dir / f"row_issues_{self.dataset_id}.parquet"
            self.row_issues.to_parquet(paths["row_issues"])
        return paths

    def to_markdown(self) -> str:
        s = self.scores
        lines = [f"# Data quality report: {self.dataset_id}", "",
                 f"Assessed {self.assessed_at} ({self.seconds}s), {self.rows:,} rows", "",
                 "## Scores", "", "| Component | Score | Weight | Checks |", "|---|---|---|---|"]
        for d in DIMENSIONS:
            score = "not evaluated" if s[d]["score"] is None else f"{s[d]['score']:.2f}"
            lines.append(f"| {d.capitalize()} | {score} | {self.weights[d]} | {s[d]['checks']} |")
        lines += [f"| **Overall** | **{s['overall']:.2f}** | | |", ""]
        if self.systematic_issues:
            lines += ["## Systematic issues (affect most rows; fix at column level)", ""]
            lines += [f"- `{c}`" for c in self.systematic_issues] + [""]
        lines += [f"Records with row-level issues (excluding systematic checks): {self.records_with_issues:,}", ""]
        lines += ["## Checks", "", "| Dimension | Check | Failed | Pass % | Severity | Source | Details |",
                  "|---|---|---|---|---|---|---|"]
        for c in self.checks:
            detail = json.dumps(c.details, default=str)
            detail = detail if len(detail) <= 120 else detail[:117] + "..."
            lines.append(f"| {c.dimension} | {c.name} | {c.failed:,} {c.unit} | {c.pass_rate * 100:.3f} | "
                         f"{c.severity} | {c.source} | {detail} |")
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


class QualityEngine:
    """
    rules (optional):
      mandatory:      [col, ...]              added to auto-detected mandatory fields
      non_negative:   [col, ...]
      ranges:         {col: [min, max]}
      allowed_values: {col: [v1, v2, ...]}
      patterns:       {col: regex}
      order:          [[earlier_col, later_col], ...]
    reference_ids:        {col: (reference_name, iterable_of_valid_ids)}  -> integrity checks
    reference_categories: {col: iterable}  valid categories learned from training data
    """

    def __init__(self, weights: dict | None = None, rules: dict | None = None,
                 reference_ids: dict | None = None, reference_categories: dict | None = None,
                 repeat_window_seconds: float = 60, systematic_threshold: float = 0.5,
                 now: pd.Timestamp | None = None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.rules = rules or {}
        self.reference_ids = reference_ids or {}
        self.reference_categories = reference_categories or {}
        self.repeat_window = repeat_window_seconds
        self.systematic_threshold = systematic_threshold
        self.now = now or pd.Timestamp.now()

    def assess(self, df: pd.DataFrame, schema: SchemaReport) -> QualityReport:
        t0 = time.time()
        logger.info("START quality assessment on %s (%d rows)", schema.dataset_id, len(df))
        roles = schema.by_role()
        cols = schema.columns
        checks: list[V.CheckResult] = []
        notes: list[str] = []

        # Parse every datetime column once.
        parsed = {c: parse_datetime_column(df[c], cols[c].role)
                  for c in roles.get("datetime", []) + roles.get("datetime_epoch", [])}
        event_cols = [c for c in roles.get("datetime", []) if cols[c].pii_type != "birth_date"]
        event_time = event_cols[0] if event_cols else None
        birth_cols = [c for c in roles.get("datetime", []) if cols[c].pii_type == "birth_date"]
        entity = (roles.get("entity_id") or [None])[0]

        # ---- completeness ----
        checks.append(V.cell_completeness(df))
        mandatory = [c for c in [schema.target, *roles.get("record_id", []), entity, event_time] if c]
        mandatory += [c for c in self.rules.get("mandatory", []) if c not in mandatory]
        if mandatory:
            checks.append(V.mandatory_fields(df, mandatory))

        # ---- validity ----
        for c in roles.get("numeric", []):
            if not pd.api.types.is_numeric_dtype(df[c]):
                checks.append(V.numeric_parseable(df[c]))
        for c in roles.get("datetime", []):
            checks.append(V.datetime_parseable(df[c], parsed[c]))
        for c in event_cols + roles.get("datetime_epoch", []):
            checks.append(V.not_in_future(c, parsed[c], self.now))
        for c in roles.get("numeric", []):
            if c in self.rules.get("non_negative", []):
                checks.append(V.in_range(df[c], 0, np.inf, source="config"))
            elif V.name_matches(c, V.AMOUNT_HINT):
                x = pd.to_numeric(df[c], errors="coerce")
                if (x.dropna() >= 0).mean() >= 0.99:
                    checks.append(V.in_range(df[c], 0, np.inf))
                else:
                    notes.append(f"`{c}` looks like an amount but is often negative; no sign rule applied")
            short = re.split(r"_", c.lower())[-1]
            if cols[c].role == "numeric" and short in GEO_RANGES and c not in self.rules.get("ranges", {}):
                checks.append(V.in_range(df[c], *GEO_RANGES[short]))
        for c, (lo, hi) in self.rules.get("ranges", {}).items():
            checks.append(V.in_range(df[c], lo, hi, source="config"))
        for c, allowed in self.rules.get("allowed_values", {}).items():
            checks.append(V.allowed_values(df[c], allowed, source="config"))
        for c, allowed in self.reference_categories.items():
            if c in df:
                checks.append(V.allowed_values(df[c], allowed, source="reference"))
        for c, pattern in self.rules.get("patterns", {}).items():
            checks.append(V.matches_pattern(df[c], pattern, source="config"))
        for c in df.columns:
            if V.name_matches(c, V.CURRENCY_HINT) and c not in self.rules.get("patterns", {}):
                checks.append(V.matches_pattern(df[c], r"[A-Z]{3}"))
        for c in roles.get("record_id", []) + roles.get("entity_id", []):
            res = V.id_format(df[c])
            if res:
                checks.append(res)
            else:
                notes.append(f"`{c}`: no dominant ID format, format check skipped")
        if schema.target:
            checks.append(V.binary_target(df[schema.target]))
        if event_time:
            for b in birth_cols:
                checks.append(V.plausible_age(b, parsed[b], parsed[event_time]))

        # ---- consistency ----
        for e in roles.get("datetime_epoch", []):
            partners = [f.split(":", 1)[1] for f in cols[e].flags if f.startswith("near_duplicate_of:")]
            partner = next((p for p in partners if p in event_cols), event_time)
            if partner:
                checks.append(V.epoch_matches_datetime(e, df[e], partner, parsed[partner]))
        if event_time:
            for b in birth_cols:
                checks.append(V.ordered(b, parsed[b], event_time, parsed[event_time]))
            for c in event_cols[1:]:
                if V.name_matches(c, V.PREVIOUS_HINT):
                    checks.append(V.ordered(c, parsed[c], event_time, parsed[event_time]))
        for earlier, later in self.rules.get("order", []):
            a = parsed.get(earlier, pd.to_numeric(df[earlier], errors="coerce"))
            b = parsed.get(later, pd.to_numeric(df[later], errors="coerce"))
            checks.append(V.ordered(earlier, a, later, b, source="config"))
        if entity:
            skip = {"record_id", "row_index", "entity_id", "target", "datetime_epoch", "constant"}
            candidates = [c for c, cs in cols.items() if cs.role not in skip and c not in event_cols]
            res = V.entity_attribute_stability(df, entity, candidates)
            if res:
                checks.append(res)

        # ---- uniqueness ----
        for c in roles.get("record_id", []):
            checks.append(V.duplicate_ids(df[c]))
        checks.append(V.duplicate_rows(df, ignore=[], name="exact_duplicate_rows"))
        id_cols = roles.get("record_id", []) + roles.get("row_index", [])
        if id_cols:
            checks.append(V.duplicate_rows(df, ignore=id_cols, name="duplicate_rows_ignoring_ids"))
        if entity and event_time:
            amounts = [c for c in roles.get("numeric", []) if V.name_matches(c, V.AMOUNT_HINT)]
            keys = amounts or roles.get("numeric", [])[:1]
            keys += [c for c in roles.get("categorical", []) if cols[c].pii_type is None
                     and cols[c].cardinality == "high"][:1]           # e.g. merchant
            if keys:
                checks.append(V.repeated_transactions(df, entity, parsed[event_time], keys, self.repeat_window))

        # ---- integrity ----
        for c, (ref_name, ids) in self.reference_ids.items():
            if c in df:
                checks.append(V.referential(df[c], ids, ref_name))
        if not any(c.dimension == "integrity" for c in checks):
            notes.append("Integrity not evaluated: no reference tables supplied "
                         "(pass reference_ids to check foreign keys); its weight is redistributed")

        # ---- classify, score, row-level issues ----
        systematic = []
        for c in checks:
            fail_rate = 1 - c.pass_rate
            c.systematic = c.unit == "rows" and fail_rate >= self.systematic_threshold
            if c.failed == 0:
                c.severity = "pass"
            elif c.name.startswith(("mandatory_fields", "duplicate_id", "target_values", "datetime_parse")):
                c.severity = "critical"
            else:
                c.severity = "warning"
            if c.systematic:
                systematic.append(c.name)
            logger.log(logging.WARNING if c.failed else logging.INFO, "%-13s %-55s failed=%s",
                       c.dimension, c.name, f"{c.failed:,}")

        scores = self._score(checks)
        row_level = {c.name: c.mask for c in checks if c.mask is not None and not c.systematic and c.failed}
        if row_level:
            row_issues = pd.DataFrame(row_level)
            row_issues["issue_count"] = row_issues.sum(axis=1)
            row_issues = row_issues[row_issues["issue_count"] > 0]
        else:
            row_issues = pd.DataFrame(columns=["issue_count"])
        report = QualityReport(dataset_id=schema.dataset_id, assessed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                               seconds=round(time.time() - t0, 1), rows=len(df), weights=self.weights,
                               scores=scores, checks=checks, records_with_issues=int(len(row_issues)),
                               systematic_issues=systematic, notes=notes, row_issues=row_issues)
        logger.info("Overall quality %.2f | %s", scores["overall"],
                    ", ".join(f"{d}={scores[d]['score']}" for d in DIMENSIONS))
        return report

    def _score(self, checks) -> dict:
        scores = {}
        for d in DIMENSIONS:
            dim = [c for c in checks if c.dimension == d]
            scores[d] = {"score": round(100 * float(np.mean([c.pass_rate for c in dim])), 4) if dim else None,
                         "checks": len(dim), "failed_checks": sum(1 for c in dim if c.failed)}
        evaluated = {d: self.weights[d] for d in DIMENSIONS if scores[d]["score"] is not None}
        total_w = sum(evaluated.values())
        scores["overall"] = round(sum(scores[d]["score"] * w for d, w in evaluated.items()) / total_w, 4)
        scores["effective_weights"] = {d: round(w / total_w, 4) for d, w in evaluated.items()}
        return scores


def assess_quality(df: pd.DataFrame, schema: SchemaReport, **kwargs) -> QualityReport:
    return QualityEngine(**kwargs).assess(df, schema)
