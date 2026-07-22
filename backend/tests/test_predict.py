"""POST /api/predict tests."""

from app.db.schemas import PredictResponse


EXPECTED_PREDICT_KEYS = {
    "id",
    "prediction",
    "confidence",
    "heatmap_url",
    "image_url",
    "report_text",
    "created_at",
}


def test_predict_with_sample_image_returns_expected_shape(client, sample_xray_file):
    response = client.post("/api/predict", files={"file": sample_xray_file})
    assert response.status_code == 200, response.text

    body = response.json()
    assert EXPECTED_PREDICT_KEYS.issubset(body.keys())

    # Validate against Pydantic schema (authoritative response shape)
    parsed = PredictResponse.model_validate(body)
    assert parsed.id >= 1
    assert parsed.prediction in {"Normal", "Pneumonia"}
    assert 0.0 <= parsed.confidence <= 1.0
    assert parsed.heatmap_url.startswith("/static/heatmaps/")
    assert parsed.image_url.startswith("/static/uploads/")
    assert isinstance(parsed.report_text, str) and len(parsed.report_text) > 0
    assert (
        "not a medical diagnosis" in parsed.report_text.lower()
        or "disclaimer" in parsed.report_text.lower()
    )


def test_predict_rejects_non_image(client):
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400
