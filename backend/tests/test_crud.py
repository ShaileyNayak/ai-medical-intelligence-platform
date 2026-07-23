"""CRUD insert/read tests for app.db.crud."""

import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.crud import (
    create_prediction,
    find_predictions_by_label,
    get_prediction,
    get_predictions_by_category,
    list_predictions,
    parse_prediction_label,
    serialize_predictions,
)
from app.db.models import Base, Prediction


def _memory_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_db_insert_and_read_via_crud():
    db = _memory_session()
    preds = [{"label": "Pneumonia", "confidence": 0.9421}]

    created = create_prediction(
        db,
        scan_type="chest_xray",
        image_path="uploads/demo.png",
        heatmap_path="heatmaps/demo_heat.png",
        predictions=preds,
        confidence=0.9421,
        report_text="Assistive report text. Not a medical diagnosis.",
    )

    assert isinstance(created, Prediction)
    assert created.id is not None
    assert created.scan_type == "chest_xray"
    assert created.image_path == "uploads/demo.png"
    assert created.heatmap_path == "heatmaps/demo_heat.png"
    assert json.loads(created.prediction_label) == preds
    assert created.confidence == 0.9421
    assert "Not a medical diagnosis" in created.report_text
    assert created.created_at is not None

    fetched = get_prediction(db, created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert parse_prediction_label(fetched.prediction_label) == preds
    assert fetched.confidence == created.confidence
    assert fetched.report_text == created.report_text

    items, total = list_predictions(db, page=1, page_size=10)
    assert total == 1
    assert len(items) == 1
    assert items[0].id == created.id


def test_db_stores_multilabel_json():
    db = _memory_session()
    preds = [
        {"label": "Pneumonia", "confidence": 0.91},
        {"label": "COVID-19", "confidence": 0.62},
    ]
    created = create_prediction(
        db,
        scan_type="chest_xray",
        image_path="m.png",
        heatmap_path="m_h.png",
        predictions=preds,
        confidence=0.91,
        report_text="multi",
    )
    assert serialize_predictions(preds) == created.prediction_label
    assert parse_prediction_label(created.prediction_label) == preds


def test_db_list_filter_by_scan_type():
    db = _memory_session()
    create_prediction(
        db,
        scan_type="chest_xray",
        image_path="a.png",
        heatmap_path="a_h.png",
        predictions=[{"label": "Normal", "confidence": 0.7}],
        confidence=0.7,
        report_text="a",
    )
    create_prediction(
        db,
        scan_type="brain_mri",
        image_path="b.png",
        heatmap_path="b_h.png",
        predictions=[{"label": "Tumor", "confidence": 0.8}],
        confidence=0.8,
        report_text="b",
    )
    items, total = list_predictions(db, page=1, page_size=10, scan_type="brain_mri")
    assert total == 1
    assert items[0].scan_type == "brain_mri"


def test_db_list_returns_newest_first():
    db = _memory_session()
    first = create_prediction(
        db,
        scan_type="chest_xray",
        image_path="a.png",
        heatmap_path="a_h.png",
        predictions=[{"label": "Normal", "confidence": 0.7}],
        confidence=0.7,
        report_text="a",
    )
    second = create_prediction(
        db,
        scan_type="chest_xray",
        image_path="b.png",
        heatmap_path="b_h.png",
        predictions=[{"label": "Pneumonia", "confidence": 0.8}],
        confidence=0.8,
        report_text="b",
    )

    items, total = list_predictions(db, page=1, page_size=10)
    assert total == 2
    assert items[0].id == second.id
    assert items[1].id == first.id


def test_get_predictions_by_category_newest_first_with_limit_offset():
    db = _memory_session()
    create_prediction(
        db,
        scan_type="chest_xray",
        image_path="c1.png",
        heatmap_path="c1_h.png",
        predictions=[{"label": "Normal", "confidence": 0.6}],
        confidence=0.6,
        report_text="c1",
    )
    brain_a = create_prediction(
        db,
        scan_type="brain_mri",
        image_path="b1.png",
        heatmap_path="b1_h.png",
        predictions=[{"label": "Tumor", "confidence": 0.9}],
        confidence=0.9,
        report_text="b1",
    )
    brain_b = create_prediction(
        db,
        scan_type="brain_mri",
        image_path="b2.png",
        heatmap_path="b2_h.png",
        predictions=[{"label": "No Tumor", "confidence": 0.85}],
        confidence=0.85,
        report_text="b2",
    )

    rows = get_predictions_by_category(db, "brain_mri", limit=10, offset=0)
    assert len(rows) == 2
    assert rows[0].id == brain_b.id
    assert rows[1].id == brain_a.id
    assert all(r.scan_type == "brain_mri" for r in rows)

    page = get_predictions_by_category(db, "brain_mri", limit=1, offset=1)
    assert len(page) == 1
    assert page[0].id == brain_a.id


def test_find_predictions_by_label_queries_json_conditions():
    db = _memory_session()
    create_prediction(
        db,
        scan_type="chest_xray",
        image_path="m.png",
        heatmap_path="m_h.png",
        predictions=[
            {"label": "Pneumonia", "confidence": 0.91},
            {"label": "COVID-19", "confidence": 0.62},
        ],
        confidence=0.91,
        report_text="multi",
    )
    create_prediction(
        db,
        scan_type="brain_mri",
        image_path="b.png",
        heatmap_path="b_h.png",
        predictions=[{"label": "Tumor", "confidence": 0.8}],
        confidence=0.8,
        report_text="brain",
    )

    hits = find_predictions_by_label(db, "COVID-19")
    assert len(hits) == 1
    assert hits[0].scan_type == "chest_xray"
    labels = [p["label"] for p in parse_prediction_label(hits[0].prediction_label)]
    assert "COVID-19" in labels

    chest_only = find_predictions_by_label(db, "pneumonia", scan_type="chest_xray")
    assert len(chest_only) == 1
