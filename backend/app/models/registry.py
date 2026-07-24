"""Map ``scan_type`` → disease module (lazy-loaded + cached)."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.models._shared import configure_torch_runtime

logger = logging.getLogger(__name__)

SCAN_TYPES = ("chest_xray", "brain_mri", "skin_lesion")


class ScanModule(Protocol):
    scan_type: str
    labels: list[str]
    model_loaded: bool

    def predict(self, image_path: str) -> dict[str, Any]: ...
    def preprocess(self, image_path: str): ...
    def make_gradcam(self): ...
    def torch_model(self): ...


# Class refs resolved lazily so importing registry does not pull all three
# packages (and their Torch graphs) until a scan_type is requested.
_MODULE_IMPORTS: dict[str, tuple[str, str]] = {
    "chest_xray": ("app.models.chest_xray.model", "ChestXrayModel"),
    "brain_mri": ("app.models.brain_mri.model", "BrainMRIModel"),
    "skin_lesion": ("app.models.skin_lesion.model", "SkinLesionModel"),
}

_instances: dict[str, ScanModule] = {}


def _resolve_module_class(scan_type: str) -> type:
    module_path, class_name = _MODULE_IMPORTS[scan_type]
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def get_model(scan_type: str) -> ScanModule:
    """
    Return the cached disease module for ``scan_type``.

    The Torch weights are loaded on first request for that type only
    (not at app startup).
    """
    configure_torch_runtime()
    key = (scan_type or "").strip().lower()
    if key not in _MODULE_IMPORTS:
        raise ValueError(
            f"Unknown scan_type '{scan_type}'. Expected one of: {', '.join(SCAN_TYPES)}"
        )
    if key not in _instances:
        logger.info("Lazy-loading model module for scan_type=%s", key)
        cls = _resolve_module_class(key)
        _instances[key] = cls()
    return _instances[key]


def list_scan_types() -> tuple[str, ...]:
    return SCAN_TYPES


def models_loaded_status() -> dict[str, bool]:
    """
    Per-module load flags for modules already in memory.

    Does **not** instantiate unloaded modules (avoids OOM on free tiers).
    Unrequested scan types report ``False`` until first prediction.
    """
    status: dict[str, bool] = {}
    for st in SCAN_TYPES:
        inst = _instances.get(st)
        status[st] = bool(inst.model_loaded) if inst is not None else False
    return status


def warmup_all() -> dict[str, bool]:
    """
    Eager-load every module. Avoid on memory-constrained hosts (e.g. Render free).

    Kept for local smoke tests that explicitly want all modules resident.
    """
    for st in SCAN_TYPES:
        get_model(st)
    return models_loaded_status()


def reset_registry() -> None:
    """Clear cached modules (tests)."""
    _instances.clear()
