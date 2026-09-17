"""Model backend registry: add a new backend by subclassing ModelAdapter and listing it here."""
from __future__ import annotations

from src.models.api_adapter import APIAdapter
from src.models.hf_adapter import HuggingFaceAdapter, NemotronAdapter
from src.models.sanity_transformer import SanityTransformerAdapter

ADAPTERS = {a.key: a for a in [SanityTransformerAdapter, HuggingFaceAdapter, NemotronAdapter, APIAdapter]}


def available_backends(hardware: dict, config: dict) -> list[dict]:
    out = []
    for key, cls in ADAPTERS.items():
        ok, message = cls.availability(hardware, config)
        out.append({"key": key, "label": cls.label, "available": ok, "message": message, "description": cls.description})
    return out


def get_adapter(key: str, config: dict):
    if key not in ADAPTERS:
        raise KeyError(f"Unknown model backend '{key}'. Available: {list(ADAPTERS)}")
    return ADAPTERS[key](config)
