from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from app.core.config import settings

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_resnet18(num_classes: int = 2, pretrained: bool = False) -> torch.nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


class InferenceService:
    """Load model once and run chest X-ray classification."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = settings.labels
        self.model: torch.nn.Module | None = None
        self.model_loaded = False
        self.transform = transforms.Compose(
            [
                transforms.Resize((settings.image_size, settings.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self._load_model()

    def _load_model(self) -> None:
        path = Path(settings.model_path)
        if not path.is_absolute():
            # Resolve relative to backend cwd and repo-relative fallbacks
            candidates = [
                path,
                Path.cwd() / path,
                Path.cwd().parent / "model" / "checkpoints" / "best_model.pt",
                Path(__file__).resolve().parents[3] / "model" / "checkpoints" / "best_model.pt",
            ]
            path = next((p for p in candidates if p.exists()), path)
        num_classes = len(self.labels)
        model = build_resnet18(num_classes=num_classes, pretrained=False)

        if path.exists():
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                if isinstance(state, dict) and any(k.startswith("module.") for k in state):
                    state = {k.replace("module.", "", 1): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
                self.model_loaded = True
                logger.info("Loaded model checkpoint from %s", path)
            except Exception:
                logger.exception("Failed loading checkpoint; using randomly initialized weights")
                self.model_loaded = False
        else:
            logger.warning("Checkpoint missing at %s — using untrained ResNet18 head", path)
            self.model_loaded = False

        model.to(self.device)
        model.eval()
        self.model = model

    def preprocess(self, image_path: str) -> tuple[torch.Tensor, Image.Image]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor, image

    @torch.no_grad()
    def predict(self, image_path: str) -> dict[str, Any]:
        assert self.model is not None
        t0 = time.perf_counter()
        tensor, _ = self.preprocess(image_path)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        class_index = int(np.argmax(probs))
        confidence = float(probs[class_index])
        label = self.labels[class_index] if class_index < len(self.labels) else str(class_index)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Inference done label=%s conf=%.4f ms=%.1f", label, confidence, elapsed_ms)
        return {
            "label": label,
            "confidence": confidence,
            "class_index": class_index,
            "probabilities": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
            "inference_ms": elapsed_ms,
            "image_path": image_path,
            "model_version": settings.model_version,
            "model_loaded": self.model_loaded,
        }

    def get_model(self) -> torch.nn.Module:
        assert self.model is not None
        return self.model


_inference_singleton: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _inference_singleton
    if _inference_singleton is None:
        _inference_singleton = InferenceService()
    return _inference_singleton
