"""Shared ResNet18 helpers used by per-module packages."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)


def build_resnet18(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet18_target_layer(model: nn.Module) -> nn.Module:
    """Last residual block — standard Grad-CAM target for ResNet18."""
    return model.layer4[-1]


def infer_num_classes_from_state(state: dict, default: int) -> int:
    for key in ("fc.weight", "module.fc.weight"):
        if key in state and hasattr(state[key], "shape"):
            return int(state[key].shape[0])
    return default


def load_resnet18_checkpoint(
    checkpoint_path: str | Path,
    *,
    num_classes: int,
    device: torch.device | str | None = None,
    class_names: list[str] | None = None,
) -> tuple[nn.Module, dict, bool]:
    """
    Load a ResNet18 checkpoint.

    Returns ``(model, meta, loaded_ok)``.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    path = Path(checkpoint_path)
    meta: dict = {"class_names": list(class_names or [])}
    loaded_ok = False

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
            state = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )

        if isinstance(state, dict) and any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        num_classes = infer_num_classes_from_state(state, num_classes)
        model = build_resnet18(num_classes=num_classes, pretrained=False)
        model.load_state_dict(state, strict=False)
        loaded_ok = True
        if "class_names" in meta and meta["class_names"]:
            meta["class_names"] = list(meta["class_names"])
        elif class_names:
            meta["class_names"] = list(class_names)[:num_classes]
    else:
        model = build_resnet18(num_classes=num_classes, pretrained=False)

    model.to(device)
    model.eval()
    meta["num_classes"] = num_classes
    return model, meta, loaded_ok


def resolve_weight_path(configured: str | Path, *fallback_names: str) -> Path:
    """Resolve a relative weights path against common backend locations."""
    path = Path(configured)
    if path.is_absolute() and path.exists():
        return path

    backend_root = Path(__file__).resolve().parents[2]
    candidates = [
        path,
        Path.cwd() / path,
        backend_root / path,
    ]
    for name in fallback_names:
        candidates.append(Path.cwd() / "model_weights" / name)
        candidates.append(backend_root / "model_weights" / name)
    candidates.append(Path.cwd() / "model_weights" / "best_model.pth")
    candidates.append(backend_root / "model_weights" / "best_model.pth")

    return next((p for p in candidates if p.exists()), path)
