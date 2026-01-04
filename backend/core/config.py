# =============================================================================
# Backend Configuration
# =============================================================================
"""
Centralized configuration for the backend API.
Uses pydantic-settings for type-safe environment variable loading.
"""

import os
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==========================================================================
    # API Settings
    # ==========================================================================
    app_name: str = Field(default="Acme CRM AI Companion API")
    app_version: str = Field(default="2.0.0")
    debug: bool = Field(default=False, description="Enable debug mode")

    # ==========================================================================
    # Server Settings
    # ==========================================================================
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=True, description="Enable auto-reload in dev")

    # ==========================================================================
    # CORS Settings
    # ==========================================================================
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:5175",
        description="Comma-separated list of allowed origins",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # ==========================================================================
    # Logging
    # ==========================================================================
    log_level: str = Field(default="INFO")
    log_requests: bool = Field(default=True, description="Log all requests")

    # ==========================================================================
    # Project Paths
    # ==========================================================================
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        # backend/core/config.py -> backend/core -> backend -> project_root
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Get the data directory (backend/data)."""
        # backend/core/config.py -> backend/core -> backend -> backend/data
        return Path(__file__).parent.parent / "data"

    model_config = {
        "env_prefix": "ACME_",
        "env_file": ".env",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


__all__ = ["Settings", "get_settings"]
