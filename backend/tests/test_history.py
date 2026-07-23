import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.crud import create_prediction, summarize_predictions_by_category
from app.db.models import Base
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


def test_history_summary_shape(client):
    response = client.get("/api/history/summary")
    assert response.status_code == 200
    body = response.json()
    for key in ("chest_xray", "brain_mri", "skin_lesion"):
        assert key in body
        assert "total" in body[key]
        assert "conditions" in body[key]
        assert "avg_confidence" in body[key]
        assert isinstance(body[key]["conditions"], dict)
        assert body[key]["total"] >= 0
        assert 0.0 <= body[key]["avg_confidence"] <= 1.0


def test_history_filter_by_scan_type(client):
    """Optional scan_type query returns only that category (or empty list)."""
    response = client.get("/api/history", params={"scan_type": "brain_mri", "page_size": 50})
    assert response.status_code == 200
    body = response.json()
    assert all(item["scan_type"] == "brain_mri" for item in body["items"])
    assert body["total"] == len(
        [i for i in body["items"]]
    ) or body["total"] >= len(body["items"])


def test_history_invalid_scan_type_returns_400(client):
    response = client.get("/api/history", params={"scan_type": "hand_xray"})
    assert response.status_code == 400
    detail = str(response.json().get("detail", "")).lower()
    assert "scan_type" in detail or "hand_xray" in detail


def test_history_limit_offset_with_scan_type(client):
    response = client.get(
        "/api/history",
        params={"scan_type": "chest_xray", "limit": 5, "offset": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["page_size"] == 5
    assert len(body["items"]) <= 5
    assert all(item["scan_type"] == "chest_xray" for item in body["items"])


def test_history_report_not_found(client):
    response = client.get("/api/history/999999/report")
    assert response.status_code == 404


def test_history_report_retrieval(client):
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        created = create_prediction(
            db,
            scan_type="brain_mri",
            image_path="uploads/report_test.png",
            heatmap_path="heatmaps/report_test_heat.png",
            predictions=[{"label": "Tumor", "confidence": 0.93}],
            confidence=0.93,
            report_text="Full stored assistive report for re-viewing.",
        )
        prediction_id = created.id
    finally:
        db.close()

    response = client.get(f"/api/history/{prediction_id}/report")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == prediction_id
    assert body["scan_type"] == "brain_mri"
    assert body["report_text"] == "Full stored assistive report for re-viewing."
    assert body["predictions"] == [{"label": "Tumor", "confidence": 0.93}]
    assert "heatmap" in body["heatmap_url"].lower() or body["heatmap_url"].endswith(
        "report_test_heat.png"
    ) or "report_test_heat" in body["heatmap_url"]


def test_summarize_predictions_by_category_counts():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    create_prediction(
        db,
        scan_type="chest_xray",
        image_path="a.png",
        heatmap_path="a_h.png",
        predictions=[
            {"label": "Pneumonia", "confidence": 0.9},
            {"label": "COVID-19", "confidence": 0.6},
        ],
        confidence=0.9,
        report_text="a",
    )
    create_prediction(
        db,
        scan_type="chest_xray",
        image_path="b.png",
        heatmap_path="b_h.png",
        predictions=[{"label": "Normal", "confidence": 0.8}],
        confidence=0.8,
        report_text="b",
    )
    create_prediction(
        db,
        scan_type="brain_mri",
        image_path="c.png",
        heatmap_path="c_h.png",
        predictions=[{"label": "Tumor", "confidence": 0.95}],
        confidence=0.95,
        report_text="c",
    )

    summary = summarize_predictions_by_category(db)
    assert summary["chest_xray"]["total"] == 2
    assert summary["chest_xray"]["conditions"]["Pneumonia"] == 1
    assert summary["chest_xray"]["conditions"]["COVID-19"] == 1
    assert summary["chest_xray"]["conditions"]["Normal"] == 1
    assert summary["chest_xray"]["avg_confidence"] == pytest.approx(0.85)
    assert summary["brain_mri"]["total"] == 1
    assert summary["brain_mri"]["conditions"] == {"Tumor": 1}
    assert summary["brain_mri"]["avg_confidence"] == pytest.approx(0.95)
    assert summary["skin_lesion"]["total"] == 0
    assert summary["skin_lesion"]["conditions"] == {}
    assert summary["skin_lesion"]["avg_confidence"] == 0.0
