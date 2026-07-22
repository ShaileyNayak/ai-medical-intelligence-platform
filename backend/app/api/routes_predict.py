from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import new_request_id
from app.db import crud
from app.db.database import get_db
from app.models.schemas import PredictionResponse
from app.services.gradcam_service import get_gradcam_service
from app.services.inference_service import get_inference_service
from app.services.llm_service import get_llm_service
from app.utils.image_utils import public_static_url, save_upload
from app.utils.validators import validate_image_upload

router = APIRouter(prefix="/api", tags=["predict"])


def _to_response(record) -> PredictionResponse:
    return PredictionResponse(
        id=record.id,
        predicted_label=record.predicted_label,
        confidence_score=record.confidence_score,
        heatmap_url=public_static_url("heatmaps", record.heatmap_path.split("/")[-1].split("\\")[-1]),
        image_url=public_static_url("uploads", record.image_filename),
        llm_report=record.llm_report,
        model_version=record.model_version,
        created_at=record.created_at,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    new_request_id()
    validate_image_upload(file)

    try:
        image_path, image_filename = await save_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    inference = get_inference_service()
    gradcam = get_gradcam_service()
    llm = get_llm_service()

    try:
        result = inference.predict(image_path)
        _, heatmap_filename, region = gradcam.generate(image_path, result)
        report = llm.generate_report(
            label=result["label"],
            confidence=result["confidence"],
            region_description=region,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction pipeline failed: {exc}") from exc

    record = crud.create_prediction(
        db,
        image_filename=image_filename,
        predicted_label=result["label"],
        confidence_score=result["confidence"],
        heatmap_path=heatmap_filename,
        llm_report=report,
        model_version=settings.model_version,
    )
    return _to_response(record)
