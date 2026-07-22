"""Download a small public chest X-ray sample set for demo training.

Uses NIH-style sample URLs / synthetic placeholders when network allows.
Creates:
  data/train/Normal
  data/train/Pneumonia
  data/test/Normal
  data/test/Pneumonia
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parents[1] / "data"

# Public-domain / openly hosted sample CXR-like images (small set)
REMOTE_SAMPLES = {
    "Normal": [
        # Wikimedia Commons chest radiograph examples (may change; fallback to synthetic)
        "https://upload.wikimedia.org/wikipedia/commons/8/8a/X-ray_of_normal_chest_by_tda.jpg",
    ],
    "Pneumonia": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Pneumonia_x-ray.jpg/640px-Pneumonia_x-ray.jpg",
    ],
}


def _synthetic(label: str, seed: int, size: int = 256) -> Image.Image:
    import random

    rng = random.Random(seed)
    img = Image.new("L", (size, size), color=20 + rng.randint(0, 20))
    draw = ImageDraw.Draw(img)
    # lung fields
    draw.ellipse((30, 40, 120, 220), fill=90 + rng.randint(0, 40))
    draw.ellipse((136, 40, 226, 220), fill=90 + rng.randint(0, 40))
    if label.lower() == "pneumonia":
        # denser opacity patches
        for _ in range(8):
            x0 = rng.randint(40, 180)
            y0 = rng.randint(60, 180)
            draw.ellipse((x0, y0, x0 + rng.randint(20, 50), y0 + rng.randint(20, 50)), fill=160)
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


def main() -> None:
    for split in ("train", "test", "samples"):
        for label in ("Normal", "Pneumonia"):
            (ROOT / split / label).mkdir(parents=True, exist_ok=True)

    # Try remote, else synthetic
    for label, urls in REMOTE_SAMPLES.items():
        images: list[Image.Image] = []
        for url in urls:
            img = _fetch(url)
            if img is not None:
                images.append(img)
        if not images:
            print(f"Using synthetic images for {label}")
            images = [_synthetic(label, seed=i) for i in range(12)]

        # Augment count with synthetics for training viability
        while len(images) < 24:
            images.append(_synthetic(label, seed=1000 + len(images)))

        for i, img in enumerate(images):
            split = "train" if i < int(len(images) * 0.8) else "test"
            path = ROOT / split / label / f"{label.lower()}_{i:03d}.png"
            img.resize((224, 224)).save(path)
            if i < 2:
                img.resize((224, 224)).save(ROOT / "samples" / label / f"sample_{i}.png")

    print(f"Dataset ready under {ROOT}")


if __name__ == "__main__":
    main()
