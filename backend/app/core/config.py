from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _normalize_database_url(url: str) -> str:
    """Render/Heroku often provide postgres:// — SQLAlchemy needs postgresql://."""
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # SQLite for local/dev; set DATABASE_URL to a postgresql://... URL for Postgres / Render
    database_url: str = "sqlite:///./predictions.db"
    model_path: str = "./model_weights/best_model.pth"
    model_version: str = "resnet18-v1.0"
    upload_dir: str = "./static/uploads"
    heatmap_dir: str = "./static/heatmaps"
    llm_provider: str = "openai"  # openai | gemini | stub
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:8501"
    log_level: str = "INFO"
    max_upload_bytes: int = 10 * 1024 * 1024
    class_names: str = "Normal,Pneumonia"
    image_size: int = 224

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_db_url(cls, value: object) -> object:
        if isinstance(value, str):
            return _normalize_database_url(value)
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def labels(self) -> list[str]:
        return [x.strip() for x in self.class_names.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
