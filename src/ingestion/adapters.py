"""Dataset adapters for structurally different transaction sources.

Adapters do not replace AutoData's schema detector. They only create the minimum
canonical fields needed when a source uses a representation the generic detector
cannot infer safely (for example PaySim's integer ``step`` clock).

Source columns are kept traceable; role overrides/hints are returned explicitly so
notebook and SaaS callers can inspect what was mapped rather than relying on hidden
renames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


PAYSIM_REQUIRED = {
    "step", "type", "amount", "nameOrig", "nameDest", "isFraud"
}
PAYSIM_BALANCE_COLUMNS = [
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
]
PAYSIM_FLAG_COLUMN = "isFlaggedFraud"
PAYSIM_EVENT_TIME = "_autodata_event_time"


@dataclass(frozen=True)
class AdapterResult:
    df: pd.DataFrame
    dataset: str
    profile: str
    target: str
    entity: str
    datetime: str
    hints: dict[str, str]
    role_overrides: dict[str, str] = field(default_factory=dict)
    dropped_columns: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def is_paysim_schema(df: pd.DataFrame) -> bool:
    """Return True when the minimum PaySim source schema is present."""
    return PAYSIM_REQUIRED.issubset(df.columns)


def adapt_paysim(
    df: pd.DataFrame,
    *,
    profile: str = "strict",
    anchor_time: str | pd.Timestamp = "2020-01-01 00:00:00",
) -> AdapterResult:
    """Adapt PaySim into AutoData's generic transaction roles.

    Profiles
    --------
    strict:
        Keeps transaction semantics needed for a deployable-style benchmark:
        ``step, type, amount, nameOrig, nameDest, isFraud`` plus a generated
        event-time column. Balance-before/after fields and ``isFlaggedFraud``
        are intentionally excluded.

    full_research:
        Also retains the four balance fields for an explicit research ablation.
        ``isFlaggedFraud`` is still excluded because its semantics are ambiguous
        and it is strongly target-adjacent in PaySim.

    Time policy
    -----------
    PaySim defines one ``step`` as one hour. We map it onto an arbitrary anchor
    timestamp. Only ordering and time deltas are used by AutoData, so the calendar
    date itself carries no business meaning. Multiple rows within the same step
    are treated as simultaneous; the point-in-time feature code therefore prevents
    them from seeing one another as history.
    """
    if profile not in {"strict", "full_research"}:
        raise ValueError("PaySim profile must be 'strict' or 'full_research'")
    missing = sorted(PAYSIM_REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"Not a PaySim-compatible frame; missing required columns: {missing}")

    out = df.copy()
    step = pd.to_numeric(out["step"], errors="coerce")
    if step.isna().any():
        bad = int(step.isna().sum())
        raise ValueError(f"PaySim 'step' must be numeric; {bad:,} rows could not be parsed")
    if (step < 0).any():
        raise ValueError("PaySim 'step' must be non-negative")

    anchor = pd.Timestamp(anchor_time)
    out[PAYSIM_EVENT_TIME] = anchor + pd.to_timedelta(step, unit="h")

    always = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]
    keep = list(always)
    dropped: list[str] = []
    if profile == "full_research":
        keep.extend([c for c in PAYSIM_BALANCE_COLUMNS if c in out.columns])
    else:
        dropped.extend([c for c in PAYSIM_BALANCE_COLUMNS if c in out.columns])
    if PAYSIM_FLAG_COLUMN in out.columns:
        dropped.append(PAYSIM_FLAG_COLUMN)
    keep.append(PAYSIM_EVENT_TIME)
    out = out[[c for c in keep if c in out.columns]].copy()

    notes = [
        "PaySim step mapped to hourly event time; anchor date is arbitrary and not a source feature.",
        "Rows sharing the same step are simultaneous and cannot see one another in history features.",
        "nameOrig is the source entity; nameDest is the counterparty/merchant-like role.",
        "isFlaggedFraud is excluded from benchmark profiles because it is target-adjacent and semantically ambiguous.",
    ]
    if profile == "strict":
        notes.append("Balance-before/after columns excluded to reduce simulator-specific shortcut/leakage risk.")
    else:
        notes.append("Balance-before/after columns retained only for explicit research comparison against strict profile.")

    return AdapterResult(
        df=out,
        dataset="PaySim",
        profile=profile,
        target="isFraud",
        entity="nameOrig",
        datetime=PAYSIM_EVENT_TIME,
        hints={"amount_column": "amount", "category_column": "type", "merchant_column": "nameDest"},
        role_overrides={"nameOrig": "ENTITY", PAYSIM_EVENT_TIME: "DATETIME"},
        dropped_columns=tuple(dropped),
        notes=tuple(notes),
        metadata={
            "time_unit": "1 hour per step",
            "anchor_time": str(anchor),
            "balance_columns_present": [c for c in PAYSIM_BALANCE_COLUMNS if c in df.columns],
            "source_columns": list(df.columns),
        },
    )
