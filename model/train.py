"""CLI training entrypoint for the chest X-ray classifier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_true")
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


def main() -> None:
    args = parse_args()
    pretrained = args.pretrained and not args.no_pretrained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    full = ChestXrayDataset(
        args.data_dir,
        split="train",
        transform=get_transforms("train", args.image_size),
    )
    if len(full) == 0:
        # try without split awareness
        full = ChestXrayDataset(args.data_dir, split="all", transform=get_transforms("train", args.image_size))
    if len(full) < 4:
        raise SystemExit(
            f"Not enough images found under {args.data_dir}. "
            "Expected folders like train/Normal, train/Pneumonia."
        )

    val_size = max(1, int(len(full) * args.val_ratio))
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size])
    # Use eval transforms for val by wrapping if possible — keep simple shared transform for demo

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
    best_path = args.checkpoint_dir / "best_model.pt"

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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  saved best checkpoint -> {best_path}")

    (args.checkpoint_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
