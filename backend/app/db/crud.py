"""CRUD helpers for the predictions table."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Prediction


def create_prediction(
    db: Session,
    *,
    image_path: str,
    heatmap_path: str,
    prediction_label: str,
    confidence: float,
    report_text: str,
) -> Prediction:
    """Insert a new prediction row and return the persisted record."""
    record = Prediction(
        image_path=image_path,
        heatmap_path=heatmap_path,
        prediction_label=prediction_label,
        confidence=confidence,
        report_text=report_text,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_predictions(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Prediction], int]:
    """
    Return a newest-first page of predictions and the total row count.

    ``page`` is 1-based.
    """
    page = max(1, page)
    page_size = max(1, page_size)

    query = db.query(Prediction)
    total = query.count()
    items = (
        query.order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def get_prediction(db: Session, prediction_id: int) -> Prediction | None:
    """Fetch a single prediction by primary key."""
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()
