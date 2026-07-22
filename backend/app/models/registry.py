"""Map ``scan_type`` → loaded disease module (model + Grad-CAM)."""

from __future__ import annotations

from typing import Any, Protocol

from app.models.brain_mri.model import BrainMRIModel
from app.models.chest_xray.model import ChestXrayModel
from app.models.skin_lesion.model import SkinLesionModel

SCAN_TYPES = ("chest_xray", "brain_mri", "skin_lesion")


class ScanModule(Protocol):
    scan_type: str
    labels: list[str]
    model_loaded: bool

    def predict(self, image_path: str) -> dict[str, Any]: ...
    def preprocess(self, image_path: str): ...
    def make_gradcam(self): ...
    def torch_model(self): ...


_MODULE_CLASSES: dict[str, type] = {
    "chest_xray": ChestXrayModel,
    "brain_mri": BrainMRIModel,
    "skin_lesion": SkinLesionModel,
}

_instances: dict[str, ScanModule] = {}


def get_model(scan_type: str) -> ScanModule:
    """
    Return the singleton disease module for ``scan_type``.

    Used by the API / service layer to run inference and Grad-CAM.
    """
    key = (scan_type or "").strip().lower()
    if key not in _MODULE_CLASSES:
        raise ValueError(
            f"Unknown scan_type '{scan_type}'. Expected one of: {', '.join(SCAN_TYPES)}"
        )
    if key not in _instances:
        _instances[key] = _MODULE_CLASSES[key]()
    return _instances[key]


def list_scan_types() -> tuple[str, ...]:
    return SCAN_TYPES


def models_loaded_status() -> dict[str, bool]:
    return {st: get_model(st).model_loaded for st in SCAN_TYPES}


def warmup_all() -> dict[str, bool]:
    """Eager-load every registered module (app startup)."""
    return models_loaded_status()


def reset_registry() -> None:
    """Clear cached modules (tests)."""
    _instances.clear()
