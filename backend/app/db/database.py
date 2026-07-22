"""Engine + session factory. SQLite by default; PostgreSQL via DATABASE_URL."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import Base


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


def init_db() -> None:
    """Create tables if they do not exist (dev-friendly; use Alembic in prod)."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
