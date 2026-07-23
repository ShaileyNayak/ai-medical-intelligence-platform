"""Grad-CAM for the chest X-ray multi-label ResNet18 classifier.

Targets the last residual block of ``layer4`` (final convolutional feature map).
For multi-label outputs, the explanation class defaults to the index with the
**highest sigmoid probability**.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.utils.image_processing import (
    as_rgb_uint8,
    blend_heatmap_overlay,
    denormalize_imagenet_tensor,
)

IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Final convolutional stage for ResNet18.

    ``model.layer4[-1]`` is the last BasicBlock; its output is the spatial
    feature map immediately before global average pooling.
    """
    if not hasattr(model, "layer4"):
        raise AttributeError("Chest X-ray Grad-CAM expects a ResNet with layer4")
    return model.layer4[-1]


class GradCAM:
    """Chest X-ray Grad-CAM with multi-label–aware class selection."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        *,
        multi_label: bool = True,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.model = model
        self.model.eval()
        self.multi_label = multi_label
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

    def _select_class_index(self, logits: torch.Tensor, class_index: int | None) -> int:
        if class_index is not None:
            return int(class_index)
        if self.multi_label:
            # Non-exclusive labels: explain the class with highest predicted probability
            probs = torch.sigmoid(logits[0])
            return int(probs.argmax().item())
        return int(logits.argmax(dim=1).item())

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
        Compute a Grad-CAM heatmap overlay as a PIL RGB image.

        When ``class_index`` is omitted and ``multi_label`` is True, uses the
        class with the highest sigmoid probability.
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("input_tensor must have shape (1, 3, H, W)")

        device = next(self.model.parameters()).device
        tensor = input_tensor.to(device).detach().clone().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        self._activations = None
        self._gradients = None

        logits = self.model(tensor)
        class_index = self._select_class_index(logits, class_index)

        score = logits[0, class_index]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        activations = self._activations.detach()
        gradients = self._gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()

        if original_image is not None:
            base_rgb = as_rgb_uint8(original_image)
        else:
            base_rgb = denormalize_imagenet_tensor(tensor, mean=self.mean, std=self.std)

        return blend_heatmap_overlay(cam, base_rgb, alpha=alpha)


def generate_gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_index: int | None = None,
    *,
    original_image: Image.Image | np.ndarray | None = None,
    alpha: float = 0.45,
    multi_label: bool = True,
) -> Image.Image:
    """Convenience wrapper returning a PIL Grad-CAM overlay."""
    with GradCAM(model, multi_label=multi_label) as cam:
        return cam.generate(
            input_tensor,
            class_index=class_index,
            original_image=original_image,
            alpha=alpha,
        )
