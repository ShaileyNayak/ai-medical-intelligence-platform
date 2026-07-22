from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import get_db
from app.db.schemas import HistoryItem, HistoryResponse
from app.utils.image_utils import public_static_url

router = APIRouter(prefix="/api", tags=["history"])


def _basename(path: str) -> str:
    return str(path).replace("\\", "/").split("/")[-1]


def record_to_history_item(record) -> HistoryItem:
    return HistoryItem(
        id=record.id,
        prediction=record.prediction_label,
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
) -> HistoryResponse:
    """Paginated prediction history from the database (newest first)."""
    items, total = crud.list_predictions(db, page=page, page_size=page_size)
    return HistoryResponse(
        items=[record_to_history_item(r) for r in items],
        total=total,
        page=page,
        page_size=page_size,
    )
