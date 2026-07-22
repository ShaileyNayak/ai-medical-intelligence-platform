from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.services.inference_service import InferenceService, get_inference_service

logger = logging.getLogger(__name__)


class GradCAMService:
    """Generate Grad-CAM heatmap overlays for ResNet18 predictions."""

    def __init__(self, inference: InferenceService | None = None) -> None:
        self.inference = inference or get_inference_service()

    def _target_layer(self, model: torch.nn.Module):
        # Last convolutional block of ResNet18
        return model.layer4[-1]

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
            overlay, region_description = self._gradcam_overlay(image_path, model, class_index)
            Image.fromarray(overlay).save(out_path)
        except Exception:
            logger.exception("Grad-CAM failed; falling back to original image copy")
            Image.open(image_path).convert("RGB").save(out_path)
            region_description = "heatmap generation fallback — original image shown"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Grad-CAM done ms=%.1f path=%s", elapsed_ms, out_path)
        return str(out_path.resolve()), filename, region_description

    def _gradcam_overlay(
        self,
        image_path: str,
        model: torch.nn.Module,
        class_index: int,
    ) -> tuple[np.ndarray, str]:
        # Prefer pytorch-grad-cam when available
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            rgb = np.array(Image.open(image_path).convert("RGB"))
            rgb_float = rgb.astype(np.float32) / 255.0
            tensor, _ = self.inference.preprocess(image_path)
            tensor = tensor.clone().detach().requires_grad_(True)

            target_layers = [self._target_layer(model)]
            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(
                    input_tensor=tensor,
                    targets=[ClassifierOutputTarget(class_index)],
                )[0]

            # Resize cam to original image size
            cam_resized = cv2.resize(grayscale_cam, (rgb.shape[1], rgb.shape[0]))
            overlay = show_cam_on_image(rgb_float, cam_resized, use_rgb=True)
            region = self._describe_region(cam_resized)
            return overlay, region
        except ImportError:
            return self._manual_gradcam(image_path, model, class_index)

    def _manual_gradcam(
        self,
        image_path: str,
        model: torch.nn.Module,
        class_index: int,
    ) -> tuple[np.ndarray, str]:
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def fwd_hook(_module, _inp, out):
            activations.append(out)

        def bwd_hook(_module, _gin, gout):
            gradients.append(gout[0])

        layer = self._target_layer(model)
        handles = [
            layer.register_forward_hook(fwd_hook),
            layer.register_full_backward_hook(bwd_hook),
        ]

        try:
            model.zero_grad(set_to_none=True)
            tensor, _ = self.inference.preprocess(image_path)
            tensor = tensor.clone().detach().requires_grad_(True)
            logits = model(tensor)
            score = logits[0, class_index]
            score.backward()

            acts = activations[0].detach()  # [1, C, H, W]
            grads = gradients[0].detach()
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            rgb = np.array(Image.open(image_path).convert("RGB"))
            cam_resized = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.uint8(0.45 * heatmap + 0.55 * rgb)
            region = self._describe_region(cam_resized)
            return overlay, region
        finally:
            for h in handles:
                h.remove()

    @staticmethod
    def _describe_region(cam: np.ndarray) -> str:
        h, w = cam.shape
        ys, xs = np.unravel_index(np.argmax(cam), cam.shape)
        vert = "upper" if ys < h / 3 else "lower" if ys > 2 * h / 3 else "mid"
        horiz = "left" if xs < w / 3 else "right" if xs > 2 * w / 3 else "central"
        intensity = float(cam.max())
        return f"{vert}-{horiz} lung field (activation intensity {intensity:.2f})"


_gradcam_singleton: GradCAMService | None = None


def get_gradcam_service() -> GradCAMService:
    global _gradcam_singleton
    if _gradcam_singleton is None:
        _gradcam_singleton = GradCAMService()
    return _gradcam_singleton
