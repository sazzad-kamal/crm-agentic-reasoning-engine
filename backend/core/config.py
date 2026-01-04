"""Centralized configuration for the backend API."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API
    app_name: str = "Acme CRM AI Companion API"
    debug: bool = False

    # CORS
    cors_origins: list[str] = ["http://localhost:5173"]

    # Logging
    log_level: str = "INFO"
    log_requests: bool = True

    @property
    def data_dir(self) -> Path:
        return Path(__file__).parent.parent / "data"

    model_config = {"env_prefix": "ACME_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
