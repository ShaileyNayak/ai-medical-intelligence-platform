from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.logging import new_request_id
from app.db import crud
from app.db.database import get_db
from app.db.schemas import PredictResponse
from app.services.gradcam_service import get_gradcam_service
from app.services.inference_service import get_inference_service
from app.services.llm_service import get_llm_service
from app.utils.image_utils import public_static_url, save_upload
from app.utils.validators import validate_image_upload

router = APIRouter(prefix="/api", tags=["predict"])


def _basename(path: str) -> str:
    return str(path).replace("\\", "/").split("/")[-1]


def record_to_predict_response(record) -> PredictResponse:
    return PredictResponse(
        id=record.id,
        prediction=record.prediction_label,
        confidence=record.confidence,
        heatmap_url=public_static_url("heatmaps", _basename(record.heatmap_path)),
        image_url=public_static_url("uploads", _basename(record.image_path)),
        report_text=record.report_text,
        created_at=record.created_at,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG/WebP)"),
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Upload an X-ray → model → Grad-CAM → LLM report → DB → JSON."""
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
        image_path=image_filename,
        heatmap_path=heatmap_filename,
        prediction_label=result["label"],
        confidence=result["confidence"],
        report_text=report,
    )
    return record_to_predict_response(record)
