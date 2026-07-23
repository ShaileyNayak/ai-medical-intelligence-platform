"""POST /api/predict — unified multi-module inference endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.logging import new_request_id
from app.db import crud
from app.db.crud import parse_prediction_label, primary_label_from_predictions
from app.db.database import get_db
from app.db.schemas import LabelConfidence, PredictResponse
from app.models.registry import SCAN_TYPES, get_model
from app.services.gradcam_service import get_gradcam_service
from app.services.llm_service import get_llm_service
from app.utils.image_utils import public_static_url, save_upload
from app.utils.validators import validate_image_upload, validate_scan_type

router = APIRouter(prefix="/api", tags=["predict"])

CHEST_POS_THRESHOLD = 0.5


def _basename(path: str) -> str:
    return str(path).replace("\\", "/").split("/")[-1]


def _format_predictions(scan_type: str, result: dict[str, Any]) -> list[LabelConfidence]:
    """
    Shape module output for the API:

    - chest_xray (multi-label): every class with confidence >= 0.5 (may be empty)
    - brain_mri / skin_lesion: a single {label, confidence}
    """
    if scan_type == "chest_xray" and result.get("multi_label", True):
        probs = result.get("probabilities") or {}
        items = [
            LabelConfidence(label=str(label), confidence=float(conf))
            for label, conf in probs.items()
            if float(conf) >= CHEST_POS_THRESHOLD
        ]
        items.sort(key=lambda x: x.confidence, reverse=True)
        return items

    return [
        LabelConfidence(
            label=str(result["label"]),
            confidence=float(result["confidence"]),
        )
    ]


def record_to_predict_response(record) -> PredictResponse:
    predictions = [
        LabelConfidence(**item) for item in parse_prediction_label(record.prediction_label)
    ]
    primary = primary_label_from_predictions(predictions)
    return PredictResponse(
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


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Medical image (JPEG/PNG/WebP)"),
    scan_type: str = Form(
        ...,
        description='One of: "chest_xray", "brain_mri", "skin_lesion"',
    ),
    db: Session = Depends(get_db),
) -> PredictResponse:
    """
    Upload an image + ``scan_type`` → registry model → Grad-CAM → LLM report → DB.

    ``prediction_label`` is stored as JSON-serialized ``[{label, confidence}, ...]``.
    """
    new_request_id()
    scan_type = validate_scan_type(scan_type)
    validate_image_upload(file)

    try:
        image_path, image_filename = await save_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        module = get_model(scan_type)
        result = module.predict(image_path)
        predictions = _format_predictions(scan_type, result)
        # Empty thresholded list → inconclusive LLM report; still store top class for history
        stored_predictions = predictions or [
            LabelConfidence(
                label=str(result["label"]),
                confidence=float(result["confidence"]),
            )
        ]
        primary_label = primary_label_from_predictions(stored_predictions)
        primary_confidence = float(stored_predictions[0].confidence)

        gradcam_payload = {
            **result,
            "label": primary_label,
            "confidence": primary_confidence,
            "class_index": int(result.get("class_index", 0)),
        }
        _, heatmap_filename, region = get_gradcam_service().generate(
            image_path, gradcam_payload, scan_type=scan_type
        )
        report = get_llm_service().generate_report(
            label=primary_label,
            confidence=primary_confidence,
            conditions=predictions,
            scan_type=scan_type,
            region_description=region,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction pipeline failed: {exc}") from exc

    record = crud.create_prediction(
        db,
        scan_type=scan_type,
        image_path=image_filename,
        heatmap_path=heatmap_filename,
        predictions=stored_predictions,
        confidence=primary_confidence,
        report_text=report,
    )
    return record_to_predict_response(record)


ALLOWED_SCAN_TYPES = SCAN_TYPES
