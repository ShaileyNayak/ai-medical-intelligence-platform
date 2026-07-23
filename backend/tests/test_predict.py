"""POST /api/predict tests — multi-disease routing."""

from __future__ import annotations

import json
from typing import Any

from app.db.crud import get_prediction, parse_prediction_label
from app.db.database import SessionLocal
from app.db.schemas import PredictResponse


EXPECTED_PREDICT_KEYS = {
    "id",
    "scan_type",
    "prediction",
    "prediction_label",
    "predictions",
    "confidence",
    "heatmap_url",
    "image_url",
    "report_text",
    "created_at",
}


class _FakeGradCAMService:
    def generate(self, image_path: str, prediction: dict[str, Any], scan_type: str = "chest_xray"):
        return "/tmp/fake_heatmap.png", "fake_heatmap.png", "mid-central region"


class _FakeMultiLabelChest:
    """Deterministic multi-label chest module for routing / DB tests."""

    scan_type = "chest_xray"

    def predict(self, image_path: str) -> dict[str, Any]:
        return {
            "scan_type": "chest_xray",
            "label": "Pneumonia",
            "confidence": 0.91,
            "class_index": 1,
            "multi_label": True,
            "probabilities": {
                "Normal": 0.12,
                "Pneumonia": 0.91,
                "COVID-19": 0.67,
                "Tuberculosis": 0.22,
            },
            "predictions": [
                {"label": "Pneumonia", "confidence": 0.91},
                {"label": "COVID-19", "confidence": 0.67},
            ],
            "model_loaded": True,
        }


def test_predict_chest_xray_returns_list_of_conditions(client, sample_xray_file):
    """scan_type=chest_xray → predictions is a list of {label, confidence}."""
    response = client.post(
        "/api/predict",
        data={"scan_type": "chest_xray"},
        files={"file": sample_xray_file},
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert EXPECTED_PREDICT_KEYS.issubset(body.keys())
    assert body["scan_type"] == "chest_xray"

    parsed = PredictResponse.model_validate(body)
    assert isinstance(parsed.predictions, list)
    assert len(parsed.predictions) >= 1
    for item in parsed.predictions:
        assert isinstance(item.label, str) and item.label
        assert 0.0 <= item.confidence <= 1.0

    stored = json.loads(parsed.prediction_label)
    assert isinstance(stored, list)
    assert stored[0]["label"] == parsed.predictions[0].label
    assert (
        "not a medical diagnosis" in parsed.report_text.lower()
        or "disclaimer" in parsed.report_text.lower()
    )


def test_predict_brain_mri_returns_single_condition(client, sample_xray_file):
    """scan_type=brain_mri → exactly one condition."""
    response = client.post(
        "/api/predict",
        data={"scan_type": "brain_mri"},
        files={"file": sample_xray_file},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scan_type"] == "brain_mri"
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["label"] in {"Tumor", "No Tumor"}
    assert 0.0 <= body["predictions"][0]["confidence"] <= 1.0
    assert json.loads(body["prediction_label"]) == body["predictions"]


def test_predict_invalid_scan_type_returns_400(client, sample_xray_file):
    response = client.post(
        "/api/predict",
        data={"scan_type": "hand_xray"},
        files={"file": sample_xray_file},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "scan_type" in detail.lower() or "hand_xray" in detail
    assert "chest_xray" in detail


def test_predict_missing_scan_type_returns_422(client, sample_xray_file):
    """Omitting the required scan_type form field → FastAPI 422 validation error."""
    response = client.post(
        "/api/predict",
        files={"file": sample_xray_file},
    )
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    # Pydantic/FastAPI validation payload mentions the missing field
    detail_text = json.dumps(body["detail"]).lower()
    assert "scan_type" in detail_text


def test_predict_chest_multilabel_db_stores_all_detected_conditions(
    client, sample_xray_file, monkeypatch
):
    """
    Multi-label chest X-ray: every condition ≥ 0.5 is persisted in
    ``prediction_label`` JSON on the DB row.
    """
    monkeypatch.setattr(
        "app.api.routes_predict.get_model",
        lambda scan_type: _FakeMultiLabelChest(),
    )
    monkeypatch.setattr(
        "app.api.routes_predict.get_gradcam_service",
        lambda: _FakeGradCAMService(),
    )

    response = client.post(
        "/api/predict",
        data={"scan_type": "chest_xray"},
        files={"file": sample_xray_file},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    api_preds = body["predictions"]
    api_labels = {p["label"] for p in api_preds}
    # Threshold 0.5 → Pneumonia (0.91) and COVID-19 (0.67); not Normal / TB
    assert api_labels == {"Pneumonia", "COVID-19"}
    assert api_preds[0]["label"] == "Pneumonia"  # ranked by confidence
    assert api_preds[0]["confidence"] == 0.91

    db = SessionLocal()
    try:
        record = get_prediction(db, body["id"])
        assert record is not None
        assert record.scan_type == "chest_xray"
        stored = parse_prediction_label(record.prediction_label)
        stored_labels = {item["label"] for item in stored}
        assert stored_labels == {"Pneumonia", "COVID-19"}
        assert stored == api_preds
        assert record.confidence == 0.91
    finally:
        db.close()


def test_predict_rejects_non_image(client):
    response = client.post(
        "/api/predict",
        data={"scan_type": "chest_xray"},
        files={"file": ("notes.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400
