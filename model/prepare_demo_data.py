"""Build a larger chest X-ray demo dataset with disjoint train/val/test splits.

Layout:
  data/train/{Normal,Pneumonia}/
  data/val/{Normal,Pneumonia}/
  data/test/{Normal,Pneumonia}/
  data/samples/{Normal,Pneumonia}/   # copies for UI demos only (not used in training)

Guarantees:
  - Unique synthetic seed per file → no identical pixel content across splits
  - Post-write MD5 check aborts if any train↔val/test overlap is detected
"""

from __future__ import annotations

import hashlib
import io
import random
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1] / "data"
LABELS = ("Normal", "Pneumonia")

# Per-class counts — total test = 2 * TEST_PER_CLASS (target 50–100 overall)
TRAIN_PER_CLASS = 120
VAL_PER_CLASS = 20
TEST_PER_CLASS = 40  # → 80 test images

REMOTE_SAMPLES = {
    "Normal": [
        "https://upload.wikimedia.org/wikipedia/commons/8/8a/X-ray_of_normal_chest_by_tda.jpg",
    ],
    "Pneumonia": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Pneumonia_x-ray.jpg/640px-Pneumonia_x-ray.jpg",
    ],
}


def _synthetic(label: str, seed: int, size: int = 256) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("L", (size, size), color=20 + rng.randint(0, 20))
    draw = ImageDraw.Draw(img)
    draw.ellipse((30, 40, 120, 220), fill=90 + rng.randint(0, 40))
    draw.ellipse((136, 40, 226, 220), fill=90 + rng.randint(0, 40))
    if label.lower() == "pneumonia":
        for _ in range(8):
            x0 = rng.randint(40, 180)
            y0 = rng.randint(60, 180)
            draw.ellipse(
                (x0, y0, x0 + rng.randint(20, 50), y0 + rng.randint(20, 50)),
                fill=160,
            )
    # Extra high-frequency noise keyed by seed → uniqueness
    for _ in range(30):
        x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
        img.putpixel((x, y), rng.randint(0, 255))
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    img = ImageEnhance.Contrast(img).enhance(1.2)
    return img.convert("RGB")


def _fetch(url: str) -> Image.Image | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "medai-demo/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        print(f"  fetch failed ({url}): {exc}")
        return None


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _clear_split_dirs() -> None:
    for split in ("train", "val", "test", "samples"):
        for label in LABELS:
            d = ROOT / split / label
            d.mkdir(parents=True, exist_ok=True)
            for p in d.glob("*"):
                if p.is_file():
                    p.unlink()


def _assert_no_leakage() -> None:
    def collect(split: str) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for label in LABELS:
            for p in (ROOT / split / label).glob("*"):
                if p.is_file():
                    out[_md5(p)] = p
        return out

    train, val, test = collect("train"), collect("val"), collect("test")
    leaks = []
    for name, other in (("val", val), ("test", test)):
        overlap = set(train) & set(other)
        for h in overlap:
            leaks.append(f"train↔{name}: {train[h].name} == {other[h].name}")
    val_test = set(val) & set(test)
    for h in val_test:
        leaks.append(f"val↔test: {val[h].name} == {test[h].name}")
    if leaks:
        raise SystemExit("DATA LEAKAGE DETECTED:\n  " + "\n  ".join(leaks))
    print(
        f"Leakage check OK — "
        f"train={len(train)} val={len(val)} test={len(test)} unique hashes, no overlaps"
    )


def main() -> None:
    _clear_split_dirs()

    # Optional real anchors (never reused across classes/splits more than once)
    anchors: dict[str, Image.Image] = {}
    for label, urls in REMOTE_SAMPLES.items():
        for url in urls:
            img = _fetch(url)
            if img is not None:
                anchors[label] = img
                break

    # Disjoint seed ranges per split so content cannot collide
    # train: 0.., val: 100_000.., test: 200_000..
    plans = (
        ("train", TRAIN_PER_CLASS, 0),
        ("val", VAL_PER_CLASS, 100_000),
        ("test", TEST_PER_CLASS, 200_000),
    )

    for label in LABELS:
        for split, count, seed_base in plans:
            for i in range(count):
                seed = seed_base + i
                # First image of train may blend in a remote anchor for realism
                if split == "train" and i == 0 and label in anchors:
                    img = anchors[label].resize((224, 224))
                else:
                    img = _synthetic(label, seed=seed).resize((224, 224))
                path = ROOT / split / label / f"{label.lower()}_{split}_{i:03d}.png"
                img.save(path)

        # UI samples: fresh seeds outside train/val/test ranges (300_000+)
        for i in range(2):
            img = _synthetic(label, seed=300_000 + i).resize((224, 224))
            img.save(ROOT / "samples" / label / f"sample_{i}.png")

    _assert_no_leakage()
    print(
        f"Dataset ready under {ROOT}\n"
        f"  per class: train={TRAIN_PER_CLASS} val={VAL_PER_CLASS} test={TEST_PER_CLASS}\n"
        f"  total test images: {TEST_PER_CLASS * len(LABELS)}"
    )


if __name__ == "__main__":
    main()
