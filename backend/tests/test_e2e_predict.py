import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_predict_end_to_end(client):
    sample = Path(__file__).resolve().parents[2] / "data" / "samples" / "Normal" / "sample_0.png"
    if not sample.exists():
        pytest.skip("sample image missing")
    with sample.open("rb") as f:
        response = client.post(
            "/api/predict",
            files={"file": ("sample_0.png", f, "image/png")},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predicted_label"] in ("Normal", "Pneumonia")
    assert 0.0 <= body["confidence_score"] <= 1.0
    assert body["heatmap_url"].startswith("/static/heatmaps/")
    assert "Disclaimer" in body["llm_report"] or "disclaimer" in body["llm_report"].lower()
