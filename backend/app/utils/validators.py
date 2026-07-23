from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.models.registry import SCAN_TYPES

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp",
    "application/octet-stream",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_scan_type(scan_type: str) -> str:
    key = (scan_type or "").strip().lower()
    if key not in SCAN_TYPES:
        allowed = ", ".join(f'"{s}"' for s in SCAN_TYPES)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid scan_type '{scan_type}'. "
                f"Allowed values are: {allowed}."
            ),
        )
    return key


def validate_image_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {suffix or '(none)'}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}",
        )


def ensure_under_size(size_bytes: int) -> None:
    if size_bytes > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max {settings.max_upload_bytes // (1024 * 1024)} MB",
        )
