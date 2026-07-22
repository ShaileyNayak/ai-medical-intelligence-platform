"""CRUD insert/read tests for app.db.crud."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.crud import create_prediction, get_prediction, list_predictions
from app.db.models import Base, Prediction


def _memory_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_db_insert_and_read_via_crud():
    db = _memory_session()

    created = create_prediction(
        db,
        image_path="uploads/demo.png",
        heatmap_path="heatmaps/demo_heat.png",
        prediction_label="Pneumonia",
        confidence=0.9421,
        report_text="Assistive report text. Not a medical diagnosis.",
    )

    assert isinstance(created, Prediction)
    assert created.id is not None
    assert created.image_path == "uploads/demo.png"
    assert created.heatmap_path == "heatmaps/demo_heat.png"
    assert created.prediction_label == "Pneumonia"
    assert created.confidence == 0.9421
    assert "Not a medical diagnosis" in created.report_text
    assert created.created_at is not None

    fetched = get_prediction(db, created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.prediction_label == "Pneumonia"
    assert fetched.confidence == created.confidence
    assert fetched.report_text == created.report_text

    items, total = list_predictions(db, page=1, page_size=10)
    assert total == 1
    assert len(items) == 1
    assert items[0].id == created.id


def test_db_list_returns_newest_first():
    db = _memory_session()
    first = create_prediction(
        db,
        image_path="a.png",
        heatmap_path="a_h.png",
        prediction_label="Normal",
        confidence=0.7,
        report_text="a",
    )
    second = create_prediction(
        db,
        image_path="b.png",
        heatmap_path="b_h.png",
        prediction_label="Pneumonia",
        confidence=0.8,
        report_text="b",
    )

    items, total = list_predictions(db, page=1, page_size=10)
    assert total == 2
    assert items[0].id == second.id
    assert items[1].id == first.id
