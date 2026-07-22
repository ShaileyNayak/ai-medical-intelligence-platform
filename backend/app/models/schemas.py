from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    id: int
    predicted_label: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    heatmap_url: str
    image_url: str
    llm_report: str
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class HistoryListResponse(BaseModel):
    items: list[PredictionResponse]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

    model_config = {"protected_namespaces": ()}


class ReportResponse(BaseModel):
    id: int
    llm_report: str
    regenerated: bool = False


class MessageResponse(BaseModel):
    message: str
