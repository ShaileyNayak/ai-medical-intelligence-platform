"""Dataset class and transforms for chest X-ray training (Normal vs Pneumonia)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Normal", "Pneumonia"]
CLASS_TO_IDX = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}


def get_transforms(split: str = "train", image_size: int = 224):
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _infer_label(path: Path) -> int | None:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "pneumonia" in parts or "pneumonia" in name:
        return CLASS_TO_IDX["pneumonia"]
    if "normal" in parts or "normal" in name:
        return CLASS_TO_IDX["normal"]
    parent = path.parent.name.lower()
    if parent in CLASS_TO_IDX:
        return CLASS_TO_IDX[parent]
    return None


class ChestXrayDataset(Dataset):
    """
    Expects either:
      root/{train,val,test}/{Normal,Pneumonia}/*.jpg
    or flat folders with class names in path.
    """

    def __init__(self, root: Path, split: str = "train", transform=None) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform or get_transforms(split)
        self.samples: list[tuple[Path, int]] = []
        self._scan()

    def _scan(self) -> None:
        split_dir = self.root / self.split
        search_root = split_dir if split_dir.exists() else self.root
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for path in sorted(search_root.rglob("*")):
            if path.suffix.lower() not in exts:
                continue
            label = _infer_label(path)
            if label is None:
                continue
            # If using split folders, only keep files under that split
            if split_dir.exists() and self.split not in path.parts:
                continue
            self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
