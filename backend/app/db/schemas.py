"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    """POST /api/predict response."""

    id: int
    prediction: str = Field(..., description="Predicted class label, e.g. Normal or Pneumonia")
    confidence: float = Field(..., ge=0.0, le=1.0)
    heatmap_url: str
    image_url: str
    report_text: str
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class HistoryItem(BaseModel):
    """Single row in GET /api/history."""

    id: int
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    heatmap_url: str
    image_url: str
    report_text: str
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


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
