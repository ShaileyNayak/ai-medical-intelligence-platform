from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import crud
from app.db.crud import parse_prediction_label, primary_label_from_predictions
from app.db.database import get_db
from app.db.schemas import (
    CategorySummary,
    HistoryItem,
    HistoryReportResponse,
    HistoryResponse,
    HistorySummaryResponse,
    LabelConfidence,
)
from app.models.registry import SCAN_TYPES
from app.utils.image_utils import public_static_url
from app.utils.validators import validate_scan_type

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


def record_to_report_response(record) -> HistoryReportResponse:
    predictions = [
        LabelConfidence(**item) for item in parse_prediction_label(record.prediction_label)
    ]
    return HistoryReportResponse(
        id=record.id,
        scan_type=record.scan_type,
        predictions=predictions,
        report_text=record.report_text or "",
        heatmap_url=public_static_url("heatmaps", _basename(record.heatmap_path)),
        image_url=public_static_url("uploads", _basename(record.image_path)),
        created_at=record.created_at,
    )


@router.get("/history/summary", response_model=HistorySummaryResponse)
def history_summary(db: Session = Depends(get_db)) -> HistorySummaryResponse:
    """
    Category-wise aggregates: total scans, condition counts, and average confidence
    for each ``scan_type`` (chest_xray / brain_mri / skin_lesion).
    """
    raw = crud.summarize_predictions_by_category(db, categories=SCAN_TYPES)
    payload = {
        key: CategorySummary(
            total=int(bucket["total"]),
            conditions=dict(bucket["conditions"]),
            avg_confidence=float(bucket["avg_confidence"]),
        )
        for key, bucket in raw.items()
        if key in SCAN_TYPES
    }
    for key in SCAN_TYPES:
        payload.setdefault(
            key,
            CategorySummary(total=0, conditions={}, avg_confidence=0.0),
        )
    return HistorySummaryResponse(**payload)


@router.get(
    "/history/{prediction_id}/report",
    response_model=HistoryReportResponse,
)
def history_report(
    prediction_id: int,
    db: Session = Depends(get_db),
) -> HistoryReportResponse:
    """
    Re-view a stored prediction report without re-running inference.

    Returns the full LLM report text, ``scan_type``, all detected conditions
    with confidence, and the Grad-CAM heatmap URL.
    """
    record = crud.get_prediction(db, prediction_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction {prediction_id} not found",
        )
    return record_to_report_response(record)


@router.get("/history", response_model=HistoryResponse)
def history(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="1-based page index"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    limit: int | None = Query(
        None,
        ge=1,
        le=100,
        description="Optional alias for page_size (pair with offset)",
    ),
    offset: int | None = Query(
        None,
        ge=0,
        description="Optional row offset (pair with limit); overrides page when set",
    ),
    scan_type: str | None = Query(
        None,
        description="Optional category filter: chest_xray | brain_mri | skin_lesion",
    ),
) -> HistoryResponse:
    """
    Paginated prediction history (newest first).

    - Omit ``scan_type`` → all categories.
    - Provide ``scan_type`` → only that category.
    - Pagination: ``page``/``page_size`` or ``limit``/``offset``.
    """
    category = None
    if scan_type is not None and str(scan_type).strip() != "":
        category = validate_scan_type(scan_type)

    use_offset = offset is not None or limit is not None
    resolved_size = limit if limit is not None else page_size
    resolved_offset = offset if offset is not None else None

    if use_offset:
        items, total = crud.list_predictions(
            db,
            limit=resolved_size,
            offset=resolved_offset if resolved_offset is not None else 0,
            scan_type=category,
        )
        off = resolved_offset if resolved_offset is not None else 0
        resolved_page = (off // resolved_size) + 1 if resolved_size else 1
    else:
        items, total = crud.list_predictions(
            db,
            page=page,
            page_size=page_size,
            scan_type=category,
        )
        resolved_page = page
        resolved_size = page_size

    return HistoryResponse(
        items=[record_to_history_item(r) for r in items],
        total=total,
        page=resolved_page,
        page_size=resolved_size,
    )
