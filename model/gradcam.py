"""Standalone Grad-CAM utility for model explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model_def import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _preprocess(image_path: Path, image_size: int = 224) -> tuple[torch.Tensor, np.ndarray]:
    rgb = np.array(Image.open(image_path).convert("RGB"))
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    tensor = tfm(Image.fromarray(rgb)).unsqueeze(0)
    return tensor, rgb


def generate_gradcam(
    model: Any,
    image_path: Path,
    target_layer: Any | None = None,
    class_index: int | None = None,
    out_path: Path | None = None,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    tensor, rgb = _preprocess(Path(image_path))
    tensor = tensor.to(device).requires_grad_(True)

    layer = target_layer or model.layer4[-1]
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_m, _i, o):
        activations.append(o)

    def bwd_hook(_m, _gi, go):
        gradients.append(go[0])

    handles = [
        layer.register_forward_hook(fwd_hook),
        layer.register_full_backward_hook(bwd_hook),
    ]
    try:
        logits = model(tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[0, class_index].backward()
        acts = activations[0].detach()
        grads = gradients[0].detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1)).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.45 * heatmap + 0.55 * rgb)
        if out_path:
            Image.fromarray(overlay).save(out_path)
        return overlay
    finally:
        for h in handles:
            h.remove()


def load_model(checkpoint: Path, num_classes: int = 2) -> torch.nn.Module:
    model = build_model(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("gradcam_out.png"))
    args = parser.parse_args()
    m = load_model(args.checkpoint)
    generate_gradcam(m, args.image, out_path=args.out)
    print(f"Wrote {args.out}")
