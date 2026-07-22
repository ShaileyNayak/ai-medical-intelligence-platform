"""
GET /api/health tests.
"""


def test_health_returns_200(client):
    response = client.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body.get("model_loaded"), bool)
    assert "model_version" in body
    assert isinstance(body.get("models_loaded"), dict)
    for key in ("chest_xray", "brain_mri", "skin_lesion"):
        assert key in body["models_loaded"]
