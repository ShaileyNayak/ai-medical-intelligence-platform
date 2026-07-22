from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import get_db
from app.models.schemas import HistoryListResponse, MessageResponse, PredictionResponse, ReportResponse
from app.services.llm_service import get_llm_service
from app.utils.image_utils import public_static_url

router = APIRouter(prefix="/api", tags=["history"])


def _to_response(record) -> PredictionResponse:
    heatmap_name = record.heatmap_path.replace("\\", "/").split("/")[-1]
    return PredictionResponse(
        id=record.id,
        predicted_label=record.predicted_label,
        confidence_score=record.confidence_score,
        heatmap_url=public_static_url("heatmaps", heatmap_name),
        image_url=public_static_url("uploads", record.image_filename),
        llm_report=record.llm_report,
        model_version=record.model_version,
        created_at=record.created_at,
    )


@router.get("/history", response_model=HistoryListResponse)
def history(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> HistoryListResponse:
    items, total = crud.list_predictions(db, page=page, page_size=page_size)
    return HistoryListResponse(
        items=[_to_response(r) for r in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/history/{prediction_id}", response_model=PredictionResponse)
def history_detail(prediction_id: int, db: Session = Depends(get_db)) -> PredictionResponse:
    record = crud.get_prediction(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return _to_response(record)


@router.delete("/history/{prediction_id}", response_model=MessageResponse)
def history_delete(prediction_id: int, db: Session = Depends(get_db)) -> MessageResponse:
    ok = crud.delete_prediction(db, prediction_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return MessageResponse(message=f"Deleted prediction {prediction_id}")


@router.get("/report/{prediction_id}", response_model=ReportResponse)
def get_or_regenerate_report(
    prediction_id: int,
    regenerate: bool = Query(False),
    db: Session = Depends(get_db),
) -> ReportResponse:
    record = crud.get_prediction(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if not regenerate:
        return ReportResponse(id=record.id, llm_report=record.llm_report, regenerated=False)

    llm = get_llm_service()
    report = llm.generate_report(
        label=record.predicted_label,
        confidence=record.confidence_score,
        region_description="previously computed Grad-CAM focus region",
    )
    updated = crud.update_report(db, prediction_id, report)
    assert updated is not None
    return ReportResponse(id=updated.id, llm_report=updated.llm_report, regenerated=True)
