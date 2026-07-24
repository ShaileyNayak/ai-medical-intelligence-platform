"""Engine + session factory. SQLite by default; PostgreSQL via DATABASE_URL."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import Base

logger = logging.getLogger(__name__)


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def create_db_engine(database_url: str | None = None) -> Engine:
    url = database_url or settings.database_url
    connect_args = {"check_same_thread": False} if _is_sqlite(url) else {}
    eng = create_engine(
        url,
        pool_pre_ping=not _is_sqlite(url),
        connect_args=connect_args,
    )

    if _is_sqlite(url):

        @event.listens_for(eng, "connect")
        def _sqlite_pragma(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return eng


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _sqlite_ensure_predictions_schema(eng: Engine) -> None:
    """
    Lightweight SQLite upgrade path for local/dev DBs that were created with
    ``create_all`` before ``scan_type`` / JSON ``prediction_label``.

    Production should use Alembic (``alembic upgrade head``).
    Skips safely when the table uses a legacy column layout that cannot be
    auto-migrated in place.
    """
    insp = inspect(eng)
    if "predictions" not in insp.get_table_names():
        return

    columns = {c["name"]: c for c in insp.get_columns("predictions")}
    with eng.begin() as conn:
        if "scan_type" not in columns:
            logger.info("Adding predictions.scan_type for local SQLite schema")
            conn.execute(
                text("ALTER TABLE predictions ADD COLUMN scan_type VARCHAR(32)")
            )
            conn.execute(
                text(
                    "UPDATE predictions SET scan_type = 'chest_xray' "
                    "WHERE scan_type IS NULL OR scan_type = ''"
                )
            )
            columns["scan_type"] = {"name": "scan_type"}

        # Ensure category index exists on older local DBs created without it
        if "scan_type" in columns:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_predictions_scan_type "
                    "ON predictions (scan_type)"
                )
            )

        # Convert legacy plain-string labels to JSON lists when needed.
        # Older schemas used predicted_label / confidence_score — leave those alone.
        if "prediction_label" not in columns or "confidence" not in columns:
            logger.warning(
                "Skipping prediction_label JSON migration — unexpected SQLite columns: %s",
                sorted(columns),
            )
            return

        rows = conn.execute(
            text("SELECT id, prediction_label, confidence FROM predictions")
        ).fetchall()
        for row in rows:
            raw = row[1]
            conf = float(row[2] or 0.0)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    continue
            except (TypeError, json.JSONDecodeError):
                pass
            payload = json.dumps(
                [{"label": str(raw), "confidence": conf}],
                separators=(",", ":"),
            )
            conn.execute(
                text("UPDATE predictions SET prediction_label = :payload WHERE id = :id"),
                {"payload": payload, "id": row[0]},
            )



def init_db() -> None:
    """Create tables if they do not exist (dev-friendly; use Alembic in prod)."""
    Base.metadata.create_all(bind=engine)
    if _is_sqlite(settings.database_url):
        _sqlite_ensure_predictions_schema(engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
