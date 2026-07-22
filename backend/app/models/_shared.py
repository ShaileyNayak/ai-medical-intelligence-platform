"""Shared ResNet18 helpers and Grad-CAM core used by per-module packages."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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
    # Legacy flat path
    candidates.append(Path.cwd() / "model_weights" / "best_model.pth")
    candidates.append(backend_root / "model_weights" / "best_model.pth")

    return next((p for p in candidates if p.exists()), path)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for a hooked target layer."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        *,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
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

        img = tensor.detach().cpu().squeeze(0).clone()
        for c in range(3):
            img[c] = img[c] * self.std[c] + self.mean[c]
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        return np.uint8(255 * img)
