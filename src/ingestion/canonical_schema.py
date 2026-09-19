"""Canonical transaction-role mapping with auditable confidence and overrides.

This layer sits between generic schema/role detection and AutoData's preprocessing stack.
It never renames the raw frame in place.  Instead it produces a mapping from canonical
roles (target/entity/time/amount/category/counterparty) to source columns, with confidence,
evidence and warnings.  Callers may supply explicit overrides; those always win and are
reported as manual confidence=1.0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.ingestion.roles import DatasetRoles
from src.ingestion.schema_detector import SchemaReport

CONFIDENCE_VALUES = {"manual": 1.0, "high": 0.95, "medium": 0.75, "low": 0.55, "none": 0.0}
CANONICAL_ROLES = ("target", "entity", "time", "amount", "category", "counterparty")


@dataclass(frozen=True)
class CanonicalField:
    role: str
    column: str | None
    confidence: float
    source: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = list(self.evidence)
        return d


@dataclass(frozen=True)
class CanonicalSchemaMapping:
    fields: dict[str, CanonicalField]
    warnings: tuple[str, ...] = ()
    overrides: dict[str, str] = field(default_factory=dict)

    def column(self, role: str) -> str | None:
        return self.fields[role].column if role in self.fields else None

    def confidence(self, role: str) -> float:
        return self.fields[role].confidence if role in self.fields else 0.0

    def ready(self, required: tuple[str, ...] = ("target",)) -> bool:
        return all(self.column(r) for r in required)

    def needs_review(self, threshold: float = 0.70) -> list[str]:
        return [r for r, f in self.fields.items() if f.column is not None and f.confidence < threshold]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "warnings": list(self.warnings),
            "overrides": dict(self.overrides),
            "needs_review": self.needs_review(),
        }


def _candidate_score(candidates: list[dict], column: str | None, default: float) -> tuple[float, tuple[str, ...]]:
    if not column:
        return 0.0, ()
    for c in candidates:
        if c.get("column") == column:
            raw = c.get("score")
            if raw is not None:
                return min(0.98, max(default, float(raw) / 6.0)), tuple(c.get("evidence") or ())
            return default, tuple([str(c.get("evidence", "detected candidate"))])
    return default, ("selected by role detector",)


def build_canonical_mapping(
    df: pd.DataFrame,
    schema: SchemaReport,
    dataset_roles: DatasetRoles,
    level_roles: Any | None = None,
    *,
    overrides: dict[str, str] | None = None,
) -> CanonicalSchemaMapping:
    """Create a canonical mapping without modifying ``df``.

    ``level_roles`` may be AutoData's preprocessing ``Roles`` object, which contributes
    amount/category/merchant(counterparty) inference.  Explicit overrides are validated
    against the source frame and always take precedence.
    """
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(CANONICAL_ROLES)
    if unknown:
        raise ValueError(f"Unknown canonical override roles: {sorted(unknown)}")
    missing = {r: c for r, c in overrides.items() if c not in df.columns}
    if missing:
        raise ValueError(f"Canonical overrides reference missing columns: {missing}")

    detected = {
        "target": dataset_roles.target,
        "entity": dataset_roles.entity,
        "time": dataset_roles.datetime,
        "amount": getattr(level_roles, "amount", None),
        "category": getattr(level_roles, "category", None),
        "counterparty": getattr(level_roles, "merchant", None),
    }
    candidate_groups = dataset_roles.candidates or {}
    fields: dict[str, CanonicalField] = {}
    warnings: list[str] = []

    for role in CANONICAL_ROLES:
        if role in overrides:
            fields[role] = CanonicalField(role, overrides[role], 1.0, "manual_override", ("explicit user/client override",))
            continue
        column = detected.get(role)
        if not column:
            fields[role] = CanonicalField(role, None, 0.0, "not_detected", ())
            continue

        if role == "target":
            conf, ev = _candidate_score(candidate_groups.get("target", []), column, 0.80)
        elif role == "entity":
            conf, ev = _candidate_score(candidate_groups.get("entity", []), column, 0.82)
        elif role == "time":
            conf, ev = _candidate_score(candidate_groups.get("datetime", []), column, 0.90)
        else:
            cs = schema.columns.get(column)
            base = CONFIDENCE_VALUES.get(getattr(cs, "confidence", "medium"), 0.75) if cs else 0.75
            conf, ev = max(0.72, min(0.90, base)), (f"selected as {role} by semantic/name/value heuristics",)
        fields[role] = CanonicalField(role, column, conf, "auto_detected", ev)

    if not fields["target"].column:
        warnings.append("No target detected: AutoData can profile/validate the data, but supervised benchmark training requires an override.")
    if not fields["time"].column:
        warnings.append("No event-time column detected: point-in-time history/sequence features will be disabled.")
    if not fields["entity"].column:
        warnings.append("No entity/account/customer column detected: behavioral history/sequence features will be disabled.")
    if not fields["amount"].column:
        warnings.append("No transaction amount/value column detected: several risk features will be unavailable.")

    return CanonicalSchemaMapping(fields=fields, warnings=tuple(warnings), overrides=overrides)
