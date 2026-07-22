from fastapi import APIRouter

from app.core.config import settings
from app.db.schemas import HealthResponse
from app.services.inference_service import get_inference_service

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness / readiness: process up and per-module weight load status."""
    svc = get_inference_service()
    loaded = svc.loaded_status()
    return HealthResponse(
        status="ok",
        model_loaded=any(loaded.values()),
        models_loaded=loaded,
        model_version=settings.model_version,
    )
