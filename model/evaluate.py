"""Evaluation metrics and confusion matrix generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, ChestXrayDataset, get_transforms
from model_def import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=Path("metrics"))
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ChestXrayDataset(args.data_dir, split=args.split, transform=get_transforms("val"))
    if len(ds) == 0:
        ds = ChestXrayDataset(args.data_dir, split="train", transform=get_transforms("val"))
    if len(ds) == 0:
        raise SystemExit(f"No images found under {args.data_dir}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    y_true, y_pred, y_prob = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs[:, 1].tolist() if probs.shape[1] > 1 else probs[:, 0].tolist())

    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred)
    report = {
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "precision": float(precision_score(y_true_a, y_pred_a, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true_a, y_pred_a, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true_a, y_pred_a, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true_a, y_pred_a, target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
        "n_samples": int(len(y_true_a)),
    }
    try:
        report["roc_auc"] = float(roc_auc_score(y_true_a, y_prob))
    except Exception:
        report["roc_auc"] = None

    cm = confusion_matrix(y_true_a, y_pred_a)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    cm_path = args.out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    (args.out_dir / "metrics_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {cm_path}")


if __name__ == "__main__":
    main()
