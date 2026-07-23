"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class LabelConfidence(BaseModel):
    """Single class score returned by /api/predict and stored in prediction_label JSON."""

    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """POST /api/predict response."""

    id: int
    scan_type: str = Field(..., description="chest_xray | brain_mri | skin_lesion")
    prediction_label: str = Field(
        ...,
        description='JSON-serialized list of {"label","confidence"} as stored in the DB',
    )
    predictions: list[LabelConfidence] = Field(
        default_factory=list,
        description=(
            "Parsed prediction_label. Chest X-ray: classes with confidence >= 0.5. "
            "Brain MRI / skin lesion: a single {label, confidence}."
        ),
    )
    prediction: str = Field(
        ...,
        description="Primary / display label (highest confidence entry)",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Primary confidence")
    heatmap_url: str
    image_url: str
    report_text: str
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class HistoryItem(BaseModel):
    """Single row in GET /api/history."""

    id: int
    scan_type: str
    prediction_label: str
    predictions: list[LabelConfidence] = Field(default_factory=list)
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    heatmap_url: str
    image_url: str
    report_text: str
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}

    @model_validator(mode="before")
    @classmethod
    def hydrate_from_orm(cls, data: object) -> object:
        """Allow constructing from ORM rows that only store prediction_label JSON."""
        if isinstance(data, dict):
            return data
        # SQLAlchemy model instance
        from app.db.crud import parse_prediction_label, primary_label_from_predictions

        preds = parse_prediction_label(getattr(data, "prediction_label", ""))
        primary = primary_label_from_predictions(preds)
        return {
            "id": data.id,
            "scan_type": getattr(data, "scan_type", "chest_xray"),
            "prediction_label": data.prediction_label,
            "predictions": preds,
            "prediction": primary,
            "confidence": data.confidence,
            "heatmap_url": getattr(data, "heatmap_url", ""),
            "image_url": getattr(data, "image_url", ""),
            "report_text": data.report_text,
            "created_at": data.created_at,
        }


class HistoryResponse(BaseModel):
    """Paginated GET /api/history response."""

    items: list[HistoryItem]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    """GET /api/health response."""

    status: str
    model_loaded: bool = False
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    model_version: str = ""

    model_config = {"protected_namespaces": ()}
