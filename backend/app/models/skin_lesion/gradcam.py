"""Grad-CAM for the skin lesion ResNet18 classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image

from app.models._shared import GradCAM as _GradCAMCore
from app.models._shared import resnet18_target_layer


def get_target_layer(model: nn.Module) -> nn.Module:
    return resnet18_target_layer(model)


class GradCAM(_GradCAMCore):
    """Skin lesion Grad-CAM — hooks ``layer4`` of ResNet18."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        super().__init__(model, target_layer or get_target_layer(model))


def generate_gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_index: int | None = None,
    *,
    original_image: Image.Image | None = None,
    alpha: float = 0.45,
) -> Image.Image:
    with GradCAM(model) as cam:
        return cam.generate(
            input_tensor,
            class_index=class_index,
            original_image=original_image,
            alpha=alpha,
        )
