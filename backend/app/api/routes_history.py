from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import crud
from app.db.crud import parse_prediction_label, primary_label_from_predictions
from app.db.database import get_db
from app.db.schemas import HistoryItem, HistoryResponse, LabelConfidence
from app.utils.image_utils import public_static_url

router = APIRouter(prefix="/api", tags=["history"])


def _basename(path: str) -> str:
    return str(path).replace("\\", "/").split("/")[-1]


def record_to_history_item(record) -> HistoryItem:
    predictions = [
        LabelConfidence(**item) for item in parse_prediction_label(record.prediction_label)
    ]
    primary = primary_label_from_predictions(predictions)
    return HistoryItem(
        id=record.id,
        scan_type=record.scan_type,
        prediction_label=record.prediction_label,
        predictions=predictions,
        prediction=primary,
        confidence=record.confidence,
        heatmap_url=public_static_url("heatmaps", _basename(record.heatmap_path)),
        image_url=public_static_url("uploads", _basename(record.image_path)),
        report_text=record.report_text,
        created_at=record.created_at,
    )


@router.get("/history", response_model=HistoryResponse)
def history(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="1-based page index"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    scan_type: str | None = Query(
        None,
        description="Optional filter: chest_xray | brain_mri | skin_lesion",
    ),
) -> HistoryResponse:
    """Paginated prediction history from the database (newest first)."""
    items, total = crud.list_predictions(
        db, page=page, page_size=page_size, scan_type=scan_type
    )
    return HistoryResponse(
        items=[record_to_history_item(r) for r in items],
        total=total,
        page=page,
        page_size=page_size,
    )
