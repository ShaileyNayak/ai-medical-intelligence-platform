"""FastAPI route package."""

from app.api.routes_health import router as health_router
from app.api.routes_history import router as history_router
from app.api.routes_predict import router as predict_router

__all__ = ["health_router", "history_router", "predict_router"]
