"""Chest X-ray module — multi-label (Normal / Pneumonia / COVID-19 / Tuberculosis)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.core.config import settings
from app.models._shared import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_resnet18_checkpoint,
    resolve_weight_path,
)
from app.models.chest_xray.gradcam import GradCAM, get_target_layer

logger = logging.getLogger(__name__)

SCAN_TYPE = "chest_xray"
# Target label set for multi-label training; binary checkpoints still load.
MULTI_LABEL_NAMES = ("Normal", "Pneumonia", "COVID-19", "Tuberculosis")
BINARY_LABEL_NAMES = ("Normal", "Pneumonia")
MULTI_LABEL_THRESHOLD = 0.5


class ChestXrayModel:
    """Model loading + inference for chest X-ray."""

    scan_type = SCAN_TYPE

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.model_loaded = False
        self.labels: list[str] = list(MULTI_LABEL_NAMES)
        self.multi_label = True
        self.image_size = settings.image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self._load()

    def _load(self) -> None:
        path = resolve_weight_path(
            settings.chest_xray_model_path,
            "chest_xray/best_model.pth",
            "best_model.pth",
        )
        model, meta, loaded = load_resnet18_checkpoint(
            path,
            num_classes=len(MULTI_LABEL_NAMES),
            device=self.device,
            class_names=list(MULTI_LABEL_NAMES),
        )
        num_classes = int(meta.get("num_classes", len(MULTI_LABEL_NAMES)))
        if num_classes == 2:
            self.labels = list(meta.get("class_names") or BINARY_LABEL_NAMES)[:2]
            self.multi_label = False
        else:
            names = list(meta.get("class_names") or MULTI_LABEL_NAMES)
            self.labels = names[:num_classes] if names else list(MULTI_LABEL_NAMES)[:num_classes]
            self.multi_label = True

        self.model = model
        self.model_loaded = loaded
        if loaded:
            logger.info("chest_xray weights loaded from %s (classes=%s)", path, self.labels)
        else:
            logger.warning("chest_xray checkpoint missing at %s — untrained head", path)

    def get_target_layer(self):
        assert self.model is not None
        return get_target_layer(self.model)

    def make_gradcam(self) -> GradCAM:
        assert self.model is not None
        return GradCAM(
            self.model,
            self.get_target_layer(),
            multi_label=self.multi_label,
        )

    def preprocess(self, image_path: str) -> tuple[torch.Tensor, Image.Image]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor, image

    @torch.inference_mode()
    def predict(self, image_path: str) -> dict[str, Any]:
        assert self.model is not None
        self.model.eval()
        t0 = time.perf_counter()
        tensor, _ = self.preprocess(image_path)
        logits = self.model(tensor)

        if self.multi_label:
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            positives = [
                {"label": self.labels[i], "confidence": float(probs[i])}
                for i in range(len(self.labels))
                if float(probs[i]) >= MULTI_LABEL_THRESHOLD
            ]
            positives.sort(key=lambda x: x["confidence"], reverse=True)
            top_i = int(np.argmax(probs))
            # Primary for Grad-CAM / DB even if no class clears the threshold
            if positives:
                primary = positives[0]
                class_index = self.labels.index(primary["label"])
            else:
                primary = {
                    "label": self.labels[top_i],
                    "confidence": float(probs[top_i]),
                }
                class_index = top_i
            probabilities = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
            label = primary["label"]
            confidence = primary["confidence"]
            # API contract: only classes above threshold (may be empty)
            predictions = positives
        else:
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            class_index = int(np.argmax(probs))
            confidence = float(probs[class_index])
            label = self.labels[class_index]
            probabilities = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
            predictions = [{"label": label, "confidence": confidence}]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "scan_type": self.scan_type,
            "label": label,
            "confidence": confidence,
            "class_index": class_index,
            "predictions": predictions,
            "probabilities": probabilities,
            "inference_ms": elapsed_ms,
            "image_path": image_path,
            "model_version": settings.model_version,
            "model_loaded": self.model_loaded,
            "multi_label": self.multi_label,
        }

    def torch_model(self) -> torch.nn.Module:
        assert self.model is not None
        return self.model
