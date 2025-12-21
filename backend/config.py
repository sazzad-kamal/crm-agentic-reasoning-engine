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
from typing import Optional

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
        default="http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
        description="Comma-separated list of allowed origins",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # ==========================================================================
    # Rate Limiting (for future use)
    # ==========================================================================
    rate_limit_enabled: bool = Field(default=False)
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window: int = Field(default=60, description="Window in seconds")

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
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.project_root / "data"

    class Config:
        env_prefix = "ACME_"
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function
def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return get_settings().debug
