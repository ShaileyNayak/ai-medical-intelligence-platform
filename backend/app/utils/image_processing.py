"""Shared image helpers for Grad-CAM overlays and related viz."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image


def as_rgb_uint8(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or ndarray to contiguous ``uint8`` RGB ``(H, W, 3)``."""
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = np.asarray(image)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        elif rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
    return np.ascontiguousarray(rgb.astype(np.uint8))


def denormalize_imagenet_tensor(
    tensor: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Denormalize a single ImageNet-normalized CHW/BCHW tensor to ``uint8`` RGB.
    """
    img = tensor.detach().cpu()
    if img.ndim == 4:
        img = img.squeeze(0)
    img = img.clone()
    for c in range(min(3, img.shape[0])):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return np.uint8(255 * img)


def blend_heatmap_overlay(
    cam: np.ndarray,
    base_image: Image.Image | np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Blend a normalized Grad-CAM map (``H×W``, values in ``[0, 1]``) onto an RGB base.

    Returns a PIL RGB image the same spatial size as ``base_image``.
    """
    base_rgb = as_rgb_uint8(base_image)
    h, w = base_rgb.shape[:2]

    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2:
        cam = np.squeeze(cam)
    if cam.ndim != 2:
        raise ValueError(f"cam must be 2-D after squeeze, got shape {cam.shape}")

    cam_min, cam_max = float(cam.min()), float(cam.max())
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = np.uint8(alpha * heatmap + (1.0 - alpha) * base_rgb)
    return Image.fromarray(overlay)
