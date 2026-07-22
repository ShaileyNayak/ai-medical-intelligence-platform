"""Model architecture: ResNet18 with custom classification head."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
