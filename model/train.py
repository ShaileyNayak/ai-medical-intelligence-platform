"""CLI training entrypoint for the chest X-ray classifier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import CLASS_NAMES, ChestXrayDataset, get_transforms
from model_def import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 chest X-ray classifier")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Used only when data/val/ is missing (carve from train; never from test)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional path to also save best checkpoint (e.g. backend model_weights)",
    )
    return parser.parse_args()


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy_from_logits(logits, labels) * bs
            n += bs
    return total_loss / max(n, 1), total_acc / max(n, 1)


class _RemappedSubset(Dataset):
    """Subset with eval transforms (avoids train augmentations on val)."""

    def __init__(self, base: ChestXrayDataset, indices: list[int], transform) -> None:
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        path, label = self.base.samples[self.indices[i]]
        from PIL import Image

        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def _load_train_val(args: argparse.Namespace):
    """
    Train/val only — the held-out ``test/`` folder is never opened here.

    Prefer explicit ``val/`` when present; otherwise carve val from ``train/``
    with a fixed generator (reproducible, still disjoint from ``test/``).
    """
    train_raw = ChestXrayDataset(
        args.data_dir,
        split="train",
        transform=get_transforms("train", args.image_size),
    )
    if len(train_raw) < 4:
        raise SystemExit(
            f"Not enough train images under {args.data_dir}/train. "
            "Run prepare_demo_data.py first. Never fall back to loading test/."
        )

    val_raw = ChestXrayDataset(
        args.data_dir,
        split="val",
        transform=get_transforms("val", args.image_size),
    )
    if len(val_raw) > 0:
        print(f"Using explicit splits: train={len(train_raw)} val={len(val_raw)} (test unused)")
        return train_raw, val_raw

    # Carve val from train only (test folder stays untouched → no leakage)
    n = len(train_raw)
    val_size = max(1, int(n * args.val_ratio))
    train_size = n - val_size
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx, val_idx = perm[:train_size], perm[train_size:]
    train_ds = _RemappedSubset(train_raw, train_idx, get_transforms("train", args.image_size))
    val_ds = _RemappedSubset(train_raw, val_idx, get_transforms("val", args.image_size))
    print(
        f"No data/val — carved from train only: "
        f"train={len(train_ds)} val={len(val_ds)} (test/ never loaded)"
    )
    return train_ds, val_ds


def main() -> None:
    args = parse_args()
    pretrained = args.pretrained and not args.no_pretrained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = _load_train_val(args)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_acc = -1.0
    best_path = args.checkpoint_dir / "best_model.pth"
    # keep legacy .pt name as alias path for older docs
    best_path_legacy = args.checkpoint_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            payload = {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "val_acc": val_acc,
                "epoch": epoch,
                "image_size": args.image_size,
            }
            torch.save(payload, best_path)
            torch.save(payload, best_path_legacy)
            if args.export:
                args.export.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, args.export)
            print(f"  saved best checkpoint -> {best_path}")

    (args.checkpoint_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
