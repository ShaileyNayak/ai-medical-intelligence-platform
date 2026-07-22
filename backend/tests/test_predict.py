import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_rejects_bad_extension(client):
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 400
