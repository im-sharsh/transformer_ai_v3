"""Transaction -> text representation for a language model.

Both experimental conditions share one template and one exclusion policy; they differ
only in the values (and derived fields) produced by the pipeline. Every choice is
recorded in the RepresentationSpec so it can be reported and reproduced.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PII_EXCLUDE_TYPES = {"person_name", "address", "birth_date", "card_number", "account_number",
                     "government_id", "email", "phone", "location"}
ID_ROLES = {"record_id", "row_index", "entity_id"}
INTERNAL_PREFIX = "_"
META_COLUMNS = {"split"}                      # bookkeeping added by the pipeline, never transaction data
CATEGORY_LABELS = {"__RARE__": "rare", "__UNSEEN__": "unseen", "UNKNOWN": "unknown"}


@dataclass
class FieldSpec:
    column: str
    key: str
    kind: str = "raw"          # raw | number2 | integer | category | day_name | yes_no | decile


@dataclass
class RepresentationSpec:
    name: str
    fields: list
    excluded: dict = field(default_factory=dict)
    header: str = "Transaction:"
    question: str = "Is this transaction fraudulent? Answer:"
    separator: str = " | "
    label_texts: dict = field(default_factory=lambda: {0: " no", 1: " yes"})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label_texts"] = {str(k): v for k, v in self.label_texts.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RepresentationSpec":
        """Rebuild a saved spec exactly (used to render new rows with an existing representation)."""
        d = dict(d)
        d["fields"] = [FieldSpec(**f) for f in d["fields"]]
        d["label_texts"] = {int(k): v for k, v in d["label_texts"].items()}
        return cls(**d)

    def render(self, df: pd.DataFrame) -> pd.Series:
        parts = [self._format(df[f.column], f) .radd(f"{f.key}=") for f in self.fields]
        body = parts[0].str.cat(parts[1:], sep=self.separator) if len(parts) > 1 else parts[0]
        return self.header + "\n" + body + "\n" + self.question

    @staticmethod
    def _format(s: pd.Series, f: FieldSpec) -> pd.Series:
        missing = s.isna()
        if f.kind == "raw":
            out = s.astype(str)                                   # exactly as stored (NaN -> 'nan')
            return out
        if f.kind == "number2":
            out = pd.to_numeric(s, errors="coerce").map(lambda v: f"{v:.2f}")
        elif f.kind == "integer":
            out = pd.to_numeric(s, errors="coerce").map(lambda v: f"{int(v)}" if pd.notna(v) else "")
        elif f.kind == "day_name":
            out = pd.to_numeric(s, errors="coerce").map(lambda v: DAY_NAMES[int(v)] if pd.notna(v) else "")
        elif f.kind == "yes_no":
            out = pd.to_numeric(s, errors="coerce").map(lambda v: "yes" if v == 1 else "no")
        elif f.kind == "decile":
            out = pd.to_numeric(s, errors="coerce").map(lambda v: f"{int(v) + 1}/10" if pd.notna(v) else "")
        else:                                                     # category
            out = s.astype(str).map(lambda v: CATEGORY_LABELS.get(v, v))
        return out.where(~missing, "missing")


# ---------------------------------------------------------------- exclusion policy
def find_quasi_identifiers(df: pd.DataFrame, entity: str, candidates: list,
                           min_stable_share: float = 0.95, min_distinct_ratio: float = 0.2) -> dict:
    """Columns fixed per entity AND with many distinct values act as a stand-in for the entity itself."""
    n_entities = df[entity].nunique()
    nunique = df.groupby(entity)[candidates].nunique(dropna=True)
    found = {}
    for c in candidates:
        stable = float((nunique[c] <= 1).mean())
        distinct = int(df[c].nunique())
        if stable >= min_stable_share and distinct >= min_distinct_ratio * n_entities:
            found[c] = (f"quasi-identifier: fixed for {stable:.0%} of {entity} values and "
                        f"{distinct} distinct values for {n_entities} entities")
    return found


def build_exclusions(roles: dict, pii: dict, target: str, quasi: dict | None = None,
                     extra: dict | None = None) -> dict:
    excluded = {target: "prediction target"}
    for role, cols in roles.items():
        if role in ID_ROLES:
            for c in cols:
                excluded[c] = f"identifier ({role})"
    for c, t in pii.items():
        if t in PII_EXCLUDE_TYPES:
            excluded.setdefault(c, f"personal data ({t})")
    for c, reason in (quasi or {}).items():
        excluded.setdefault(c, reason)
    for c, reason in (extra or {}).items():
        excluded.setdefault(c, reason)
    return excluded


def raw_spec(raw_columns: list, excluded: dict) -> RepresentationSpec:
    """Minimal formatting: every allowed original column, values as stored, original order."""
    fields = [FieldSpec(c, c, "raw") for c in raw_columns
              if not c.startswith(INTERNAL_PREFIX) and c not in META_COLUMNS and c not in excluded]
    return RepresentationSpec("raw", fields, excluded={c: r for c, r in excluded.items() if c in raw_columns})


def processed_spec(processed_columns: list, excluded: dict, pii: dict, event_time: str,
                   categorical: list, numeric: list, boolean: list, time_fields: list) -> RepresentationSpec:
    """Cleaned base columns + train-fitted derived fields. Exact timestamps replaced by time fields."""
    fields, notes = [], dict(excluded)
    for c in processed_columns:
        if c.startswith(INTERNAL_PREFIX) or c in META_COLUMNS or c in excluded or "__" in c or c in time_fields:
            continue
        if c == event_time:
            notes[c] = "replaced by hour / day_of_week / is_weekend / month"
        elif c in categorical or c in boolean:
            fields.append(FieldSpec(c, c, "category"))
        elif c in numeric:
            fields.append(FieldSpec(c, c, "number2"))
    kinds = {"hour": "integer", "day_of_week": "day_name", "is_weekend": "yes_no", "month": "integer"}
    for t in time_fields:
        if t in processed_columns:
            fields.append(FieldSpec(t, t, kinds.get(t, "integer")))
    for c in processed_columns:
        m = re.fullmatch(r"(.+)__bin", c)
        if m and m.group(1) in numeric and pii.get(m.group(1)) not in PII_EXCLUDE_TYPES:
            fields.append(FieldSpec(c, f"{m.group(1)}_decile", "decile"))
        elif m:
            notes[c] = f"bucket of personal data column {m.group(1)}"
        elif re.fullmatch(r".+__(robust|log)", c):
            notes[c] = "scaled copy for tabular baselines; the decile is used in text instead"
        elif re.fullmatch(r".+__was_missing", c):
            fields.append(FieldSpec(c, c.replace("__was_missing", "_missing"), "yes_no"))
    return RepresentationSpec("processed", fields, excluded=notes)


# ---------------------------------------------------------------- dataset building
def target_word_share(prompts: pd.Series, spec: RepresentationSpec, words=("fraud",)) -> float:
    body = prompts.str.replace(re.escape(spec.question), "", regex=True)
    return float(body.str.contains("|".join(words), case=False).mean())


def build_text_dataset(df: pd.DataFrame, spec: RepresentationSpec, target: str, tokenizer=None,
                       max_tokens: int = 256) -> tuple[pd.DataFrame, dict]:
    for f in spec.fields:
        assert f.column != target and target not in f.column, f"target leaks into field {f.column}"
    prompts = spec.render(df)
    labels = pd.to_numeric(df[target]).astype(int)
    out = pd.DataFrame({"_row_id": df["_row_id"].values, "prompt": prompts.values, "label": labels.values,
                        "label_text": labels.map(spec.label_texts).values})
    for c in ["_weight", "_slice_unseen_entity", "split"]:
        if c in df:
            out[c] = df[c].values
    stats = {"rows": len(out), "fields": len(spec.fields), "positives": int(labels.sum()),
             "target_word_share": round(target_word_share(prompts, spec), 4)}
    if tokenizer is not None:
        counts = [len(ids) for ids in tokenizer(prompts.tolist(), add_special_tokens=True)["input_ids"]]
        label_lengths = {k: len(tokenizer(v, add_special_tokens=False)["input_ids"]) for k, v in spec.label_texts.items()}
        out["prompt_tokens"] = counts
        c = np.array(counts)
        stats.update({"tokens_mean": round(float(c.mean()), 1), "tokens_p95": int(np.percentile(c, 95)),
                      "tokens_max": int(c.max()), "over_max_tokens": int((c + 1 > max_tokens).sum()),
                      "label_token_lengths": label_lengths})
    return out, stats


def save_representation(root: Path, spec: RepresentationSpec, datasets: dict, stats: dict, extra: dict) -> None:
    d = root / spec.name
    d.mkdir(parents=True, exist_ok=True)
    for split, frame in datasets.items():
        frame.to_parquet(d / f"{split}.parquet", index=False)
    (d / "spec.json").write_text(json.dumps({"spec": spec.to_dict(), "stats": stats, **extra}, indent=2, default=str))
