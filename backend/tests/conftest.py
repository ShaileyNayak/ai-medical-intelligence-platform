"""Shared pytest fixtures for API and DB tests."""

from __future__ import annotations

import io
import os
from pathlib import Path

# Isolate tests from legacy local DBs (e.g. medical_ai.db with old columns).
_TEST_DB = Path(__file__).resolve().parent / "_pytest_predictions.db"
if _TEST_DB.exists():
    _TEST_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture()
def client():
    """FastAPI test client with lifespan (DB init; models stay lazy)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_xray_bytes() -> bytes:
    """
    Minimal valid PNG chest-X-ray stand-in for upload tests.

    Prefer a real sample from the repo when present; otherwise synthesize one.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "samples" / "Normal" / "sample_0.png",
        Path(__file__).resolve().parents[3] / "data" / "samples" / "Normal" / "sample_0.png",
        Path(__file__).resolve().parent / "fixtures" / "sample_xray.png",
    ]
    for path in candidates:
        if path.exists():
            return path.read_bytes()

    img = Image.new("RGB", (224, 224), color=(30, 30, 30))
    # Simple lung-field ellipses so Grad-CAM / preprocess have structure
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 50, 100, 180), fill=(90, 90, 90))
    draw.ellipse((124, 50, 184, 180), fill=(90, 90, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_xray_file(sample_xray_bytes):
    """Multipart-ready tuple for TestClient uploads."""
    return ("sample_xray.png", sample_xray_bytes, "image/png")
