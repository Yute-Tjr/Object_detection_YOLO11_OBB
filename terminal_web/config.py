from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by the API and inference worker."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    storage_root: Path = Field(validation_alias="INSPECTION_STORAGE_ROOT")
    detector_weights: Path
    label3_classifier_weights: Path
    label5_classifier_weights: Path
    detection_device: str = "cpu"
    classification_device: str = "cpu"
    detection_imgsz: int = 1280
    classification_imgsz: int = 224
    detection_confidence: float = 0.25
    max_images_per_task: int = 100
    task_max_attempts: int = 2
    worker_heartbeat_timeout_seconds: int = 120

    @field_validator("database_url")
    @classmethod
    def require_postgresql(cls, value: str) -> str:
        if not value.startswith("postgresql+"):
            raise ValueError("DATABASE_URL must use PostgreSQL")
        return value

    @field_validator(
        "storage_root",
        "detector_weights",
        "label3_classifier_weights",
        "label5_classifier_weights",
    )
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
