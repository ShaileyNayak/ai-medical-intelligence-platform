"""Grad-CAM for the brain MRI binary ResNet18 classifier.

Targets the last residual block of ``layer4`` (final convolutional feature map).
Explains the predicted class (argmax over softmax / logits) by default.
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
    Final convolutional stage for ResNet18 used by the brain MRI module.

    ``model.layer4[-1]`` is the last BasicBlock before global average pooling.
    """
    if not hasattr(model, "layer4"):
        raise AttributeError("Brain MRI Grad-CAM expects a ResNet with layer4")
    return model.layer4[-1]


class GradCAM:
    """Brain MRI Grad-CAM — binary Tumor / No Tumor explanations."""

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
        """Compute a Grad-CAM heatmap overlay as a PIL RGB image."""
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
) -> Image.Image:
    """Convenience wrapper returning a PIL Grad-CAM overlay."""
    with GradCAM(model) as cam:
        return cam.generate(
            input_tensor,
            class_index=class_index,
            original_image=original_image,
            alpha=alpha,
        )
