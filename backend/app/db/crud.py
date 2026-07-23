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
    Parse ``prediction_label`` JSON into a list of ``{label, confidence}``.

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

    ``scan_type`` is the primary category. ``predictions`` is JSON-serialized into
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


def get_predictions_by_category(
    db: Session,
    scan_type: str,
    limit: int = 20,
    offset: int = 0,
) -> list[Prediction]:
    """
    Return predictions for one category (``scan_type``), newest first.

    Uses the ``ix_predictions_scan_type`` index for filtering.
    """
    limit = max(1, int(limit))
    offset = max(0, int(offset))
    category = (scan_type or "").strip().lower()

    return (
        db.query(Prediction)
        .filter(Prediction.scan_type == category)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def list_predictions(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    limit: int | None = None,
    offset: int | None = None,
    scan_type: str | None = None,
) -> tuple[list[Prediction], int]:
    """
    Return a newest-first page of predictions and the total row count.

    Pagination: use ``limit``/``offset`` when provided; otherwise ``page``/
    ``page_size`` (1-based). Optional ``scan_type`` filters by primary category.
    """
    if limit is not None or offset is not None:
        size = max(1, int(limit if limit is not None else page_size))
        off = max(0, int(offset or 0))
    else:
        page = max(1, page)
        size = max(1, page_size)
        off = (page - 1) * size

    query = db.query(Prediction)
    if scan_type:
        query = query.filter(Prediction.scan_type == scan_type.strip().lower())
    total = query.count()
    items = (
        query.order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .offset(off)
        .limit(size)
        .all()
    )
    return items, total


def find_predictions_by_label(
    db: Session,
    label: str,
    *,
    scan_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    case_insensitive: bool = True,
) -> list[Prediction]:
    """
    Query rows whose ``prediction_label`` JSON list contains a matching condition.

    Matching is done against the stored JSON text (portable across SQLite and
    PostgreSQL). Results are newest-first; optional ``scan_type`` narrows by
    category first (indexed).
    """
    needle = (label or "").strip()
    if not needle:
        return []

    limit = max(1, int(limit))
    offset = max(0, int(offset))
    query = db.query(Prediction)
    if scan_type:
        query = query.filter(Prediction.scan_type == scan_type.strip().lower())

    # JSON is stored compact: {"label":"Pneumonia","confidence":0.9}
    # Also match pretty-printed variants with a space after the colon.
    pattern = f'%\"label\":\"{needle}\"%'
    pattern_spaced = f'%\"label\": \"{needle}\"%'

    from sqlalchemy import or_

    if case_insensitive:
        query = query.filter(
            or_(
                Prediction.prediction_label.ilike(pattern),
                Prediction.prediction_label.ilike(pattern_spaced),
            )
        )
    else:
        query = query.filter(
            or_(
                Prediction.prediction_label.like(pattern),
                Prediction.prediction_label.like(pattern_spaced),
            )
        )

    candidates = (
        query.order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .offset(offset)
        .limit(limit * 3)  # over-fetch then exact-match on parsed JSON
        .all()
    )

    needle_cmp = needle.lower() if case_insensitive else needle
    matched: list[Prediction] = []
    for row in candidates:
        for item in parse_prediction_label(row.prediction_label):
            lbl = str(item.get("label", ""))
            if case_insensitive:
                if lbl.lower() == needle_cmp:
                    matched.append(row)
                    break
            elif lbl == needle_cmp:
                matched.append(row)
                break
        if len(matched) >= limit:
            break
    return matched


def get_prediction(db: Session, prediction_id: int) -> Prediction | None:
    """Fetch a single prediction by primary key."""
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()


def summarize_predictions_by_category(
    db: Session,
    categories: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Aggregate history by primary category (``scan_type``).

    For each category returns:
    - ``total``: number of prediction rows
    - ``conditions``: count of each label appearing in ``prediction_label`` JSON
    - ``avg_confidence``: mean of row ``confidence`` (0.0 when empty)
    """
    from app.models.registry import SCAN_TYPES

    keys = tuple(categories) if categories else SCAN_TYPES
    summary: dict[str, dict[str, Any]] = {
        key: {"total": 0, "conditions": {}, "avg_confidence": 0.0} for key in keys
    }

    rows = db.query(Prediction).all()
    conf_sums: dict[str, float] = {key: 0.0 for key in keys}
    conf_counts: dict[str, int] = {key: 0 for key in keys}

    for row in rows:
        category = (row.scan_type or "").strip().lower()
        if category not in summary:
            summary[category] = {"total": 0, "conditions": {}, "avg_confidence": 0.0}
            conf_sums[category] = 0.0
            conf_counts[category] = 0

        bucket = summary[category]
        bucket["total"] += 1

        conf = float(row.confidence or 0.0)
        conf_sums[category] += conf
        conf_counts[category] += 1

        for item in parse_prediction_label(row.prediction_label):
            label = str(item.get("label") or "Unknown")
            conditions: dict[str, int] = bucket["conditions"]
            conditions[label] = conditions.get(label, 0) + 1

    for category, bucket in summary.items():
        n = conf_counts.get(category, 0)
        bucket["avg_confidence"] = round(conf_sums[category] / n, 6) if n else 0.0

    return summary
