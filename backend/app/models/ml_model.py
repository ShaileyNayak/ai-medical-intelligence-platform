"""ResNet18 chest X-ray classifier used for inference and Grad-CAM."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)
DEFAULT_CLASS_NAMES = ("Normal", "Pneumonia")


def build_resnet18(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """Build ResNet18 with a custom classification head."""
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block — standard Grad-CAM target for ResNet18."""
    return model.layer4[-1]


def load_resnet18_checkpoint(
    checkpoint_path: str | Path,
    *,
    num_classes: int = 2,
    device: torch.device | str | None = None,
) -> tuple[nn.Module, dict]:
    """
    Load a training checkpoint into a ResNet18.

    Supports dicts with ``model_state_dict`` (notebook / train.py format)
    or raw ``state_dict`` tensors.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    path = Path(checkpoint_path)
    model = build_resnet18(num_classes=num_classes, pretrained=False)

    meta: dict = {"class_names": list(DEFAULT_CLASS_NAMES)}
    if path.exists():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            meta = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            state = checkpoint
        else:
            state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        if isinstance(state, dict) and any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        if "class_names" in meta and meta["class_names"]:
            meta["class_names"] = list(meta["class_names"])

    model.to(device)
    model.eval()
    return model, meta
