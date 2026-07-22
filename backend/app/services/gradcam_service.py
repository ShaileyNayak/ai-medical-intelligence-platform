from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from app.core.config import settings
from app.models.gradcam import GradCAM
from app.services.inference_service import InferenceService, get_inference_service

logger = logging.getLogger(__name__)


class GradCAMService:
    """Generate Grad-CAM heatmap overlays via ``app.models.gradcam``."""

    def __init__(self, inference: InferenceService | None = None) -> None:
        self.inference = inference or get_inference_service()

    def generate(self, image_path: str, prediction: dict[str, Any]) -> tuple[str, str, str]:
        """
        Returns (absolute_heatmap_path, heatmap_filename, region_description).
        """
        t0 = time.perf_counter()
        out_dir = Path(settings.heatmap_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid4().hex}_heatmap.png"
        out_path = out_dir / filename

        model = self.inference.get_model()
        class_index = int(prediction.get("class_index", 0))
        region_description = "model focus region unavailable"

        try:
            tensor, pil_image = self.inference.preprocess(image_path)
            with GradCAM(model) as cam:
                overlay = cam.generate(
                    tensor,
                    class_index=class_index,
                    original_image=pil_image,
                )
            overlay.save(out_path)
            # Approximate focus region from a second grayscale pass description
            region_description = self._describe_from_overlay(overlay)
        except Exception:
            logger.exception("Grad-CAM failed; falling back to original image copy")
            Image.open(image_path).convert("RGB").save(out_path)
            region_description = "heatmap generation fallback — original image shown"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Grad-CAM done ms=%.1f path=%s", elapsed_ms, out_path)
        return str(out_path.resolve()), filename, region_description

    @staticmethod
    def _describe_from_overlay(overlay: Image.Image) -> str:
        """Heuristic region label from warm (jet) pixels in the overlay."""
        arr = np.asarray(overlay.convert("RGB"), dtype=np.float32)
        # Jet colormap: high activation → strong red/yellow channels
        heat = arr[..., 0] - 0.5 * arr[..., 2]
        heat = np.clip(heat, 0, None)
        if float(heat.max()) < 1e-6:
            return "diffuse activation across the lung fields"
        ys, xs = np.unravel_index(int(np.argmax(heat)), heat.shape)
        h, w = heat.shape
        vert = "upper" if ys < h / 3 else "lower" if ys > 2 * h / 3 else "mid"
        horiz = "left" if xs < w / 3 else "right" if xs > 2 * w / 3 else "central"
        intensity = float(heat.max() / (heat.max() + 1e-8))
        return f"{vert}-{horiz} lung field (activation intensity {intensity:.2f})"


_gradcam_singleton: GradCAMService | None = None


def get_gradcam_service() -> GradCAMService:
    global _gradcam_singleton
    if _gradcam_singleton is None:
        _gradcam_singleton = GradCAMService()
    return _gradcam_singleton
