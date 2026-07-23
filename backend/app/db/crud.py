"""CRUD helpers for the predictions table."""

from __future__ import annotations

import json
from typing import Any, Sequence

from sqlalchemy.orm import Session

from app.db.models import Prediction


def serialize_predictions(predictions: Sequence[Any]) -> str:
    """Serialize label/confidence items to the DB ``prediction_label`` JSON text."""
    payload: list[dict[str, Any]] = []
    for item in predictions:
        if hasattr(item, "model_dump"):
            data = item.model_dump()
        elif isinstance(item, dict):
            data = item
        else:
            raise TypeError(f"Unsupported prediction item type: {type(item)!r}")
        payload.append(
            {
                "label": str(data["label"]),
                "confidence": float(data["confidence"]),
            }
        )
    return json.dumps(payload, separators=(",", ":"))


def parse_prediction_label(raw: str | None) -> list[dict[str, Any]]:
    """
    Parse ``prediction_label`` JSON.

    Supports legacy plain-string rows (pre-migration) by wrapping them as a
    single-item list with confidence 0.0.
    """
    if raw is None or str(raw).strip() == "":
        return []
    text = str(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [{"label": text, "confidence": 0.0}]

    if isinstance(data, list):
        out: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "label": str(item.get("label", "Unknown")),
                    "confidence": float(item.get("confidence", 0.0)),
                }
            )
        return out

    if isinstance(data, str):
        return [{"label": data, "confidence": 0.0}]

    return [{"label": str(data), "confidence": 0.0}]


def primary_label_from_predictions(predictions: Sequence[Any]) -> str:
    """Highest-confidence label (first item if already sorted)."""
    if not predictions:
        return "Unknown"
    first = predictions[0]
    if hasattr(first, "label"):
        return str(first.label)
    return str(first.get("label", "Unknown"))


def create_prediction(
    db: Session,
    *,
    scan_type: str,
    image_path: str,
    heatmap_path: str,
    predictions: Sequence[Any],
    confidence: float,
    report_text: str,
    prediction_label: str | None = None,
) -> Prediction:
    """
    Insert a prediction row.

    ``predictions`` is the canonical list of ``{label, confidence}`` (or
    Pydantic ``LabelConfidence`` models). It is JSON-serialized into
    ``prediction_label`` unless an explicit serialized string is passed.
    """
    stored_label = (
        prediction_label
        if prediction_label is not None
        else serialize_predictions(predictions)
    )
    record = Prediction(
        scan_type=scan_type,
        image_path=image_path,
        heatmap_path=heatmap_path,
        prediction_label=stored_label,
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
    scan_type: str | None = None,
) -> tuple[list[Prediction], int]:
    """
    Return a newest-first page of predictions and the total row count.

    ``page`` is 1-based. Optional ``scan_type`` filters the result set.
    """
    page = max(1, page)
    page_size = max(1, page_size)

    query = db.query(Prediction)
    if scan_type:
        query = query.filter(Prediction.scan_type == scan_type)
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
