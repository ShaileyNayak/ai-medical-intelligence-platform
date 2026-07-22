"""GET /api/health tests."""


def test_health_returns_200(client):
    response = client.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body.get("model_loaded"), bool)
    assert "model_version" in body
