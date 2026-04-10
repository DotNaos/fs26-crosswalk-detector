"""Crosswalk detector package.

Keep package import lightweight.
Heavy helpers from ``pilot`` are loaded lazily so importing API modules does not
pay the full data-pipeline startup cost.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "bootstrap_local_pilot",
    "build_pilot_summary",
    "build_full_capped_dataset",
    "crosswalk_score",
    "derive_split_targets",
    "export_balanced_seed_set",
    "fetch_wmts_search_tiles",
    "fetch_wmts_debug_tiles",
    "image_label_pair_count",
    "load_pilot_config",
    "raw_size_mb",
]

_DATASET_EXPORTS = {"image_label_pair_count"}
_PILOT_EXPORTS = {
    "bootstrap_local_pilot",
    "build_pilot_summary",
    "build_full_capped_dataset",
    "crosswalk_score",
    "derive_split_targets",
    "export_balanced_seed_set",
    "fetch_wmts_search_tiles",
    "fetch_wmts_debug_tiles",
    "load_pilot_config",
    "raw_size_mb",
}


def __getattr__(name: str) -> Any:
    if name in _DATASET_EXPORTS:
        module = import_module(".dataset", __name__)
        return getattr(module, name)
    if name in _PILOT_EXPORTS:
        module = import_module(".pilot", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
