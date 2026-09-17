"""Text representation of prepared rows for language-model backends."""
from __future__ import annotations

import pandas as pd

from src.representation.transaction_formatter import FieldSpec, RepresentationSpec


def spec_for(prepared, question: str = "Is this transaction fraudulent? Answer:") -> RepresentationSpec:
    fields = [FieldSpec(c, c, "number2") for c in prepared.numeric] + [FieldSpec(c, c, "category") for c in prepared.categorical]
    return RepresentationSpec(name=prepared.level, fields=fields, question=question)


def prompts(frame: pd.DataFrame, spec: RepresentationSpec) -> pd.Series:
    return spec.render(frame)
