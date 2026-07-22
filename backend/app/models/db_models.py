"""Compatibility shim — ORM models live in ``app.db.models``."""

from app.db.models import Base, Prediction

__all__ = ["Base", "Prediction"]
