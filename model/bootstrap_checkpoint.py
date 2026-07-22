"""Create an initial checkpoint (ImageNet-pretrained ResNet18 head) for demo boot.

Prefer training with train.py on real data. This script ensures deploy works
even before a full training run completes.
"""

from __future__ import annotations

from pathlib import Path

import torch

from model_def import build_model

CLASS_NAMES = ["Normal", "Pneumonia"]


def main() -> None:
    out = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(num_classes=2, pretrained=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "val_acc": None,
            "note": "bootstrap ImageNet-pretrained head; replace after train.py",
        },
        out,
    )
    print(f"Wrote bootstrap checkpoint -> {out}")


if __name__ == "__main__":
    main()
