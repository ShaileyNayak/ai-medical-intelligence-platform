"""Brain MRI module — binary Tumor / No Tumor."""

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
from app.models.brain_mri.gradcam import GradCAM, get_target_layer

logger = logging.getLogger(__name__)

SCAN_TYPE = "brain_mri"
LABEL_NAMES = ("Tumor", "No Tumor")


class BrainMRIModel:
    """Model loading + inference for brain MRI."""

    scan_type = SCAN_TYPE

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.model_loaded = False
        self.labels: list[str] = list(LABEL_NAMES)
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
            settings.brain_mri_model_path,
            "brain_mri/best_model.pth",
        )
        model, meta, loaded = load_resnet18_checkpoint(
            path,
            num_classes=len(LABEL_NAMES),
            device=self.device,
            class_names=list(LABEL_NAMES),
        )
        names = list(meta.get("class_names") or LABEL_NAMES)
        self.labels = names[:2] if names else list(LABEL_NAMES)
        self.model = model
        self.model_loaded = loaded
        if loaded:
            logger.info("brain_mri weights loaded from %s", path)
        else:
            logger.warning("brain_mri checkpoint missing at %s — untrained head", path)

    def get_target_layer(self):
        assert self.model is not None
        return get_target_layer(self.model)

    def make_gradcam(self) -> GradCAM:
        assert self.model is not None
        return GradCAM(self.model, self.get_target_layer())

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
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        class_index = int(np.argmax(probs))
        confidence = float(probs[class_index])
        label = self.labels[class_index]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "scan_type": self.scan_type,
            "label": label,
            "confidence": confidence,
            "class_index": class_index,
            "predictions": [{"label": label, "confidence": confidence}],
            "probabilities": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
            "inference_ms": elapsed_ms,
            "image_path": image_path,
            "model_version": settings.model_version,
            "model_loaded": self.model_loaded,
            "multi_label": False,
        }

    def torch_model(self) -> torch.nn.Module:
        assert self.model is not None
        return self.model
