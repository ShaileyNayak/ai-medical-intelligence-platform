"""SQLAlchemy ORM models for the persistence layer."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """
    Stored inference result for one uploaded study.

    ``prediction_label`` holds a JSON-serialized list of
    ``{"label": str, "confidence": float}`` objects (length 1 for binary modules,
    1–N for multi-label chest X-ray).
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    heatmap_path: Mapped[str] = mapped_column(String(512), nullable=False)
    prediction_label: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    report_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
