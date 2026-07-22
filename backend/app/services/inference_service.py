from __future__ import annotations

import logging
from typing import Any

from app.models.registry import get_model, models_loaded_status, warmup_all

logger = logging.getLogger(__name__)


class InferenceService:
    """Thin facade over ``app.models.registry.get_model`` for the API layer."""

    def predict(self, image_path: str, scan_type: str = "chest_xray") -> dict[str, Any]:
        module = get_model(scan_type)
        return module.predict(image_path)

    def preprocess(self, image_path: str, scan_type: str = "chest_xray"):
        return get_model(scan_type).preprocess(image_path)

    def get_torch_model(self, scan_type: str = "chest_xray"):
        return get_model(scan_type).torch_model()

    # Backward-compatible alias used by older Grad-CAM service code
    def get_model(self, scan_type: str = "chest_xray"):
        return self.get_torch_model(scan_type)

    @property
    def model_loaded(self) -> bool:
        status = models_loaded_status()
        return bool(status.get("chest_xray"))

    def loaded_status(self) -> dict[str, bool]:
        return models_loaded_status()


_inference_singleton: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _inference_singleton
    if _inference_singleton is None:
        warmup_all()
        _inference_singleton = InferenceService()
    return _inference_singleton
