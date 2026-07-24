#!/usr/bin/env python3
"""
Evaluate each disease module on its held-out test split.

Writes:
  backend/model_weights/<scan_type>/metrics_report.json

If a checkpoint or test set is missing, builds a small synthetic dataset and
(optionally) trains a short demo checkpoint so metrics can still be produced.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
WEIGHTS_ROOT = BACKEND_ROOT / "model_weights"
DATA_ROOT = REPO_ROOT / "data"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ModuleSpec:
    scan_type: str
    class_names: tuple[str, ...]
    data_dirs: tuple[Path, ...]
    weight_path: Path
    folder_aliases: dict[str, str]


SPECS: dict[str, ModuleSpec] = {
    "chest_xray": ModuleSpec(
        scan_type="chest_xray",
        class_names=("Normal", "Pneumonia"),
        data_dirs=(DATA_ROOT / "chest_xray", DATA_ROOT),
        weight_path=WEIGHTS_ROOT / "chest_xray" / "best_model.pth",
        folder_aliases={
            "normal": "Normal",
            "pneumonia": "Pneumonia",
        },
    ),
    "brain_mri": ModuleSpec(
        scan_type="brain_mri",
        class_names=("Tumor", "No Tumor"),
        data_dirs=(DATA_ROOT / "brain_mri",),
        weight_path=WEIGHTS_ROOT / "brain_mri" / "best_model.pth",
        folder_aliases={
            "tumor": "Tumor",
            "no tumor": "No Tumor",
            "no_tumor": "No Tumor",
            "notumor": "No Tumor",
            "no-tumor": "No Tumor",
        },
    ),
    "skin_lesion": ModuleSpec(
        scan_type="skin_lesion",
        class_names=("Malignant", "Benign"),
        data_dirs=(DATA_ROOT / "skin_lesion",),
        weight_path=WEIGHTS_ROOT / "skin_lesion" / "best_model.pth",
        folder_aliases={
            "malignant": "Malignant",
            "benign": "Benign",
        },
    ),
}


def _canonical_label(name: str, spec: ModuleSpec) -> str | None:
    raw = name.strip()
    if raw in spec.class_names:
        return raw
    key = raw.lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    for alias, canonical in spec.folder_aliases.items():
        if key == alias or key == alias.replace(" ", ""):
            return canonical
    # also match lowercase class names
    for c in spec.class_names:
        if key == c.lower():
            return c
    return None


def collect_split_samples(spec: ModuleSpec, split: str) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for root in spec.data_dirs:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_name = _canonical_label(path.parent.name, spec)
            if label_name is None:
                continue
            samples.append((path, spec.class_names.index(label_name)))
        if samples:
            break
    return samples


def eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ImageFolderSplit(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def build_resnet18(num_classes: int) -> nn.Module:
    try:
        model = models.resnet18(weights=None)
    except Exception:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = build_resnet18(num_classes=num_classes)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    # Adapt head if checkpoint class count differs
    if isinstance(state, dict) and "fc.weight" in state:
        ckpt_classes = int(state["fc.weight"].shape[0])
        if ckpt_classes != num_classes:
            model = build_resnet18(num_classes=ckpt_classes)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _synthetic_image(spec: ModuleSpec, label: str, seed: int, size: int = 224) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), color=(15 + rng.randint(0, 10),) * 3)
    draw = ImageDraw.Draw(img)

    if spec.scan_type == "chest_xray":
        draw.ellipse((30, 40, 100, 190), fill=(90 + rng.randint(0, 30),) * 3)
        draw.ellipse((124, 40, 194, 190), fill=(90 + rng.randint(0, 30),) * 3)
        if label == "Pneumonia":
            for _ in range(6):
                x0, y0 = rng.randint(40, 160), rng.randint(60, 150)
                draw.ellipse((x0, y0, x0 + 35, y0 + 35), fill=(150,) * 3)
    elif spec.scan_type == "brain_mri":
        draw.ellipse((40, 30, 184, 200), fill=(70 + rng.randint(0, 20),) * 3)
        if label == "Tumor":
            x0, y0 = rng.randint(70, 130), rng.randint(70, 130)
            draw.ellipse((x0, y0, x0 + 40, y0 + 40), fill=(200,) * 3)
    else:  # skin_lesion
        draw.ellipse((50, 50, 174, 174), fill=(180, 140, 120))
        if label == "Malignant":
            draw.ellipse((80, 80, 140, 150), fill=(40, 20, 20))
            for _ in range(8):
                x, y = rng.randint(70, 150), rng.randint(70, 150)
                draw.point((x, y), fill=(10, 10, 10))
        else:
            draw.ellipse((90, 90, 130, 130), fill=(160, 110, 90))

    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    img = ImageEnhance.Contrast(img).enhance(1.15)
    return img


def bootstrap_dataset(spec: ModuleSpec, per_class: int = 24) -> Path:
    """Create train/test synthetic images under the primary data dir."""
    root = spec.data_dirs[0]
    print(f"[{spec.scan_type}] Bootstrapping synthetic dataset under {root}")
    for split, n in (("train", int(per_class * 0.75)), ("test", per_class - int(per_class * 0.75))):
        for label in spec.class_names:
            out = root / split / label
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                seed = hash((spec.scan_type, split, label, i)) % (2**31)
                img = _synthetic_image(spec, label, seed)
                safe = label.lower().replace(" ", "_")
                img.save(out / f"{safe}_{i:03d}.png")
    return root


def train_demo_checkpoint(
    spec: ModuleSpec,
    train_samples: list[tuple[Path, int]],
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 8,
) -> Path:
    print(f"[{spec.scan_type}] Training demo checkpoint ({epochs} epochs, n={len(train_samples)})…")
    ds = ImageFolderSplit(train_samples, eval_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = build_resnet18(num_classes=len(spec.class_names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        n = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += images.size(0)
        print(
            f"  epoch {epoch}/{epochs} loss={total_loss / max(n, 1):.4f} "
            f"acc={correct / max(n, 1):.3f}"
        )

    spec.weight_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": list(spec.class_names),
        "num_classes": len(spec.class_names),
        "image_size": 224,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "scan_type": spec.scan_type,
    }
    torch.save(payload, spec.weight_path)
    print(f"[{spec.scan_type}] Saved {spec.weight_path}")
    return spec.weight_path


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    samples: list[tuple[Path, int]],
    device: torch.device,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(ImageFolderSplit(samples, eval_transform()), batch_size=batch_size, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: tuple[str, ...],
) -> dict:
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_samples": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=list(class_names),
            zero_division=0,
            output_dict=True,
        ),
    }
    try:
        if y_prob.shape[1] == 2:
            # Positive class = index 0 for Tumor/Malignant; use class-1 prob for Normal/Pneumonia style
            # Standard binary AUC uses probability of the positive class.
            # Use max-class convention: AUC of class index 1 vs rest for Pneumonia-style,
            # but for Tumor (0) / No Tumor (1) notebooks used Tumor as positive (index 0).
            pos_index = 0 if class_names[0] in {"Tumor", "Malignant"} else 1
            metrics["auc"] = float(roc_auc_score(y_true == pos_index, y_prob[:, pos_index]))
        else:
            metrics["auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
    except Exception:
        metrics["auc"] = float("nan")
    return metrics


def ensure_ready(
    spec: ModuleSpec,
    device: torch.device,
    *,
    allow_train: bool,
    epochs: int,
) -> tuple[list[tuple[Path, int]], Path]:
    test_samples = collect_split_samples(spec, "test")
    train_samples = collect_split_samples(spec, "train")

    if not test_samples:
        bootstrap_dataset(spec)
        test_samples = collect_split_samples(spec, "test")
        train_samples = collect_split_samples(spec, "train")

    if not test_samples:
        raise RuntimeError(f"[{spec.scan_type}] No test images found after bootstrap")

    weight_path = spec.weight_path
    if not weight_path.exists():
        if not allow_train:
            raise FileNotFoundError(
                f"[{spec.scan_type}] Missing checkpoint {weight_path}. "
                "Re-run with --train-missing to create a demo model."
            )
        if not train_samples:
            bootstrap_dataset(spec)
            train_samples = collect_split_samples(spec, "train")
            test_samples = collect_split_samples(spec, "test")
        train_demo_checkpoint(spec, train_samples, device, epochs=epochs)
    return test_samples, weight_path


def evaluate_module(
    scan_type: str,
    device: torch.device,
    *,
    allow_train: bool,
    epochs: int,
) -> dict:
    spec = SPECS[scan_type]
    t0 = time.time()
    test_samples, weight_path = ensure_ready(
        spec, device, allow_train=allow_train, epochs=epochs
    )
    model = load_checkpoint(weight_path, num_classes=len(spec.class_names), device=device)
    # If checkpoint has different class count, rebuild class_names from fc
    n_out = model.fc.out_features
    class_names = spec.class_names
    if n_out != len(class_names):
        # Fall back to first n labels / generic names
        class_names = tuple(list(spec.class_names)[:n_out]) or tuple(
            f"class_{i}" for i in range(n_out)
        )

    y_true, y_pred, y_prob = run_inference(model, test_samples, device)
    # Filter labels if class count mismatch
    if y_prob.shape[1] != len(class_names):
        class_names = tuple(f"class_{i}" for i in range(y_prob.shape[1]))

    test_metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    report = {
        "scan_type": scan_type,
        "best_model_path": str(weight_path.resolve()),
        "test_split": "test",
        "class_names": list(class_names),
        "n_test": int(len(test_samples)),
        "seconds": round(time.time() - t0, 2),
        "test_metrics": test_metrics,
        # flat convenience fields
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "auc": test_metrics["auc"],
        "f1": test_metrics["f1"],
    }

    out_path = weight_path.parent / "metrics_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== {scan_type} ===")
    print(f"checkpoint: {weight_path}")
    print(f"test n={report['n_test']}")
    print(f"accuracy:  {report['accuracy']:.4f}")
    print(f"precision: {report['precision']:.4f}")
    print(f"recall:    {report['recall']:.4f}")
    print(f"AUC:       {report['auc']:.4f}")
    print(f"wrote:     {out_path}")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all disease modules on test split")
    p.add_argument(
        "--modules",
        nargs="+",
        default=list(SPECS.keys()),
        choices=list(SPECS.keys()),
    )
    p.add_argument(
        "--train-missing",
        action="store_true",
        help="Train a short demo checkpoint when best_model.pth is missing",
    )
    p.add_argument("--epochs", type=int, default=6, help="Demo train epochs when needed")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Modules: {', '.join(args.modules)}")

    summary = {}
    for name in args.modules:
        try:
            summary[name] = evaluate_module(
                name,
                device,
                allow_train=args.train_missing,
                epochs=args.epochs,
            )
        except Exception as exc:
            print(f"[{name}] FAILED: {exc}", file=sys.stderr)
            summary[name] = {"error": str(exc)}

    print("\n===== SUMMARY =====")
    for name, rep in summary.items():
        if "error" in rep:
            print(f"{name}: ERROR — {rep['error']}")
        else:
            print(
                f"{name}: acc={rep['accuracy']:.3f}  "
                f"P={rep['precision']:.3f}  R={rep['recall']:.3f}  "
                f"AUC={rep['auc']:.3f}"
            )
    return 0 if all("error" not in v for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
