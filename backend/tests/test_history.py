import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_history_list(client):
    response = client.get("/api/history")
    assert response.status_code == 200
    body = response.json()
    assert "items" in body
    assert "total" in body
    assert "page" in body
    assert "page_size" in body


def test_history_pagination(client):
    response = client.get("/api/history", params={"page": 1, "page_size": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 5
    assert len(body["items"]) <= 5
