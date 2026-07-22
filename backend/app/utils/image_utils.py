from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings


async def save_upload(file: UploadFile) -> tuple[str, str]:
    """Save upload; returns (absolute_path, filename)."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    original = Path(file.filename or "upload.png")
    suffix = original.suffix.lower() or ".png"
    filename = f"{uuid4().hex}{suffix}"
    dest = upload_dir / filename
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise ValueError(f"File exceeds max size of {settings.max_upload_bytes} bytes")
    dest.write_bytes(content)
    return str(dest.resolve()), filename


def public_static_url(subdir: str, filename: str) -> str:
    return f"/static/{subdir}/{filename}"
