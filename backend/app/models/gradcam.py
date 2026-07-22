"""Grad-CAM explainability for the ResNet18 model defined in ``ml_model``."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.models.ml_model import IMAGENET_MEAN, IMAGENET_STD, get_target_layer


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for ResNet18.

    Hooks the last convolutional block, computes a class-specific heatmap,
    and returns a blended RGB overlay as a ``PIL.Image``.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        *,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or get_target_layer(model)
        self.mean = tuple(mean)
        self.std = tuple(std)

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []

        self._handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, _module, _inputs, output) -> None:
        self._activations = output

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @torch.enable_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        class_index: int | None = None,
        *,
        original_image: Image.Image | np.ndarray | None = None,
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Compute a Grad-CAM overlay for ``input_tensor``.

        Parameters
        ----------
        input_tensor:
            Preprocessed batch tensor shaped ``(1, 3, H, W)`` (ImageNet-normalized).
        class_index:
            Class to explain. Defaults to the model's predicted class.
        original_image:
            Optional RGB image / array for overlay. If omitted, the tensor is
            denormalized for visualization.
        alpha:
            Heatmap blend factor (0–1).

        Returns
        -------
        PIL.Image.Image
            RGB overlay highlighting regions that most influenced the prediction.
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("input_tensor must have shape (1, 3, H, W)")

        device = next(self.model.parameters()).device
        tensor = input_tensor.to(device).detach().clone().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        self._activations = None
        self._gradients = None

        logits = self.model(tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        score = logits[0, class_index]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        activations = self._activations.detach()  # (1, C, h, w)
        gradients = self._gradients.detach()  # (1, C, h, w)

        # Channel weights via global-average-pooled gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = float(cam.min()), float(cam.max())
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        base_rgb = self._resolve_base_image(tensor, original_image)
        h, w = base_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = np.uint8(alpha * heatmap + (1.0 - alpha) * base_rgb)
        return Image.fromarray(overlay)

    def _resolve_base_image(
        self,
        tensor: torch.Tensor,
        original_image: Image.Image | np.ndarray | None,
    ) -> np.ndarray:
        if original_image is not None:
            if isinstance(original_image, Image.Image):
                rgb = np.array(original_image.convert("RGB"))
            else:
                rgb = np.asarray(original_image)
                if rgb.ndim == 2:
                    rgb = np.stack([rgb] * 3, axis=-1)
                elif rgb.shape[-1] == 4:
                    rgb = rgb[..., :3]
            return rgb.astype(np.uint8)

        # Denormalize ImageNet tensor → uint8 RGB at model input resolution
        img = tensor.detach().cpu().squeeze(0).clone()
        for c in range(3):
            img[c] = img[c] * self.std[c] + self.mean[c]
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        return np.uint8(255 * img)


def generate_gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_index: int | None = None,
    *,
    original_image: Image.Image | np.ndarray | None = None,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Convenience wrapper: Grad-CAM heatmap overlay as a PIL image.

    ``input_tensor`` must be a preprocessed ``(1, 3, H, W)`` batch.
    """
    with GradCAM(model) as cam:
        return cam.generate(
            input_tensor,
            class_index=class_index,
            original_image=original_image,
            alpha=alpha,
        )
