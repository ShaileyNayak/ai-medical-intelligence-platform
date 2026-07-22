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

    # Per-module weight paths (one .pth under model_weights/<scan_type>/)
    chest_xray_model_path: str = "./model_weights/chest_xray/best_model.pth"
    brain_mri_model_path: str = "./model_weights/brain_mri/best_model.pth"
    skin_lesion_model_path: str = "./model_weights/skin_lesion/best_model.pth"
    # Legacy alias — defaults to chest module
    model_path: str = "./model_weights/chest_xray/best_model.pth"

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
    class_names: str = "Normal,Pneumonia,COVID-19,Tuberculosis"
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

    def model_path_for(self, scan_type: str) -> str:
        mapping = {
            "chest_xray": self.chest_xray_model_path,
            "brain_mri": self.brain_mri_model_path,
            "skin_lesion": self.skin_lesion_model_path,
        }
        return mapping.get(scan_type, self.model_path)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
