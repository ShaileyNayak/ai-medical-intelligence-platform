from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import HealthResponse
from app.services.inference_service import get_inference_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = get_inference_service()
    return HealthResponse(
        status="ok",
        model_loaded=svc.model_loaded,
        model_version=settings.model_version,
    )
