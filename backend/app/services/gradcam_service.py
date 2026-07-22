from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from app.core.config import settings
from app.models.registry import get_model

logger = logging.getLogger(__name__)


class GradCAMService:
    """Generate Grad-CAM overlays via the module registered for ``scan_type``."""

    def generate(
        self,
        image_path: str,
        prediction: dict[str, Any],
        scan_type: str = "chest_xray",
    ) -> tuple[str, str, str]:
        """
        Returns (absolute_heatmap_path, heatmap_filename, region_description).
        """
        t0 = time.perf_counter()
        out_dir = Path(settings.heatmap_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid4().hex}_heatmap.png"
        out_path = out_dir / filename

        module = get_model(scan_type)
        class_index = int(prediction.get("class_index", 0))
        region_description = "model focus region unavailable"

        try:
            tensor, pil_image = module.preprocess(image_path)
            with module.make_gradcam() as cam:
                overlay = cam.generate(
                    tensor,
                    class_index=class_index,
                    original_image=pil_image,
                )
            overlay.save(out_path)
            region_description = self._describe_from_overlay(overlay, scan_type=scan_type)
        except Exception:
            logger.exception("Grad-CAM failed; falling back to original image copy")
            Image.open(image_path).convert("RGB").save(out_path)
            region_description = "heatmap generation fallback — original image shown"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Grad-CAM done scan_type=%s ms=%.1f path=%s", scan_type, elapsed_ms, out_path)
        return str(out_path.resolve()), filename, region_description

    @staticmethod
    def _describe_from_overlay(overlay: Image.Image, scan_type: str = "chest_xray") -> str:
        """Heuristic region label from warm (jet) pixels in the overlay."""
        arr = np.asarray(overlay.convert("RGB"), dtype=np.float32)
        heat = arr[..., 0] - 0.5 * arr[..., 2]
        heat = np.clip(heat, 0, None)
        tissue = {
            "chest_xray": "lung field",
            "brain_mri": "brain region",
            "skin_lesion": "lesion area",
        }.get(scan_type, "region")
        if float(heat.max()) < 1e-6:
            return f"diffuse activation across the {tissue}"
        ys, xs = np.unravel_index(int(np.argmax(heat)), heat.shape)
        h, w = heat.shape
        vert = "upper" if ys < h / 3 else "lower" if ys > 2 * h / 3 else "mid"
        horiz = "left" if xs < w / 3 else "right" if xs > 2 * w / 3 else "central"
        intensity = float(heat.max() / (heat.max() + 1e-8))
        return f"{vert}-{horiz} {tissue} (activation intensity {intensity:.2f})"


_gradcam_singleton: GradCAMService | None = None


def get_gradcam_service() -> GradCAMService:
    global _gradcam_singleton
    if _gradcam_singleton is None:
        _gradcam_singleton = GradCAMService()
    return _gradcam_singleton
