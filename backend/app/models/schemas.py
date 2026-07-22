"""Backward-compatible re-exports — prefer ``app.db.schemas``."""

from app.db.schemas import (  # noqa: F401
    HealthResponse,
    HistoryItem,
    HistoryResponse,
    PredictResponse,
)

PredictionResponse = PredictResponse
HistoryListResponse = HistoryResponse
