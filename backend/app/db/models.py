"""SQLAlchemy ORM models for the persistence layer."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """
    Stored inference result for one uploaded study.

    Category
    --------
    ``scan_type`` is the primary category / imaging module:
    ``chest_xray`` | ``brain_mri`` | ``skin_lesion``.
    It is indexed (``ix_predictions_scan_type``) for fast category filtering.

    Conditions
    ----------
    ``prediction_label`` stores a JSON-serialized list of
    ``{"label": str, "confidence": float}`` objects (length 1 for binary modules,
    1–N for multi-label chest X-ray). Use ``parse_prediction_label`` /
    ``find_predictions_by_label`` in ``crud`` to query condition labels.
    """

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_scan_type", "scan_type"),
        Index("ix_predictions_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Primary category — chest_xray | brain_mri | skin_lesion
    scan_type: Mapped[str] = mapped_column(String(32), nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    heatmap_path: Mapped[str] = mapped_column(String(512), nullable=False)
    # JSON text: [{"label": "...", "confidence": 0.0}, ...] — query via crud helpers
    prediction_label: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    report_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
