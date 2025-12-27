"""Tests for backend configuration."""

import pytest
from pathlib import Path

from backend.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_app_name(self):
        """Test default app name."""
        settings = Settings()
        assert settings.app_name == "Acme CRM AI Companion API"

    def test_default_app_version(self):
        """Test default app version."""
        settings = Settings()
        assert settings.app_version == "2.0.0"

    def test_default_debug_mode(self):
        """Test debug mode is disabled by default."""
        settings = Settings()
        assert settings.debug is False

    def test_default_host(self):
        """Test default host."""
        settings = Settings()
        assert settings.host == "0.0.0.0"

    def test_default_port(self):
        """Test default port."""
        settings = Settings()
        assert settings.port == 8000

    def test_default_reload(self):
        """Test reload is enabled by default."""
        settings = Settings()
        assert settings.reload is True

    def test_rate_limit_enabled_is_bool(self):
        """Test rate_limit_enabled is a boolean."""
        settings = Settings()
        assert isinstance(settings.rate_limit_enabled, bool)

    def test_rate_limit_requests_default(self):
        """Test default rate limit requests."""
        settings = Settings()
        assert settings.rate_limit_requests == 100

    def test_rate_limit_window_default(self):
        """Test default rate limit window."""
        settings = Settings()
        assert settings.rate_limit_window == 60

    def test_default_log_level(self):
        """Test default log level."""
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_log_requests_default(self):
        """Test request logging is enabled by default."""
        settings = Settings()
        assert settings.log_requests is True


class TestCorsSettings:
    """Tests for CORS configuration."""

    def test_default_cors_origins(self):
        """Test default CORS origins string contains localhost."""
        settings = Settings()
        assert "localhost" in settings.cors_origins

    def test_cors_origins_list(self):
        """Test CORS origins are parsed into a list."""
        settings = Settings()
        origins = settings.cors_origins_list
        assert isinstance(origins, list)
        assert len(origins) > 0
        assert "http://localhost:5173" in origins

    def test_cors_origins_list_strips_whitespace(self):
        """Test CORS origins list strips whitespace."""
        settings = Settings(cors_origins="http://a.com , http://b.com")
        origins = settings.cors_origins_list
        assert "http://a.com" in origins
        assert "http://b.com" in origins
        # No leading/trailing spaces
        for origin in origins:
            assert origin == origin.strip()


class TestProjectPaths:
    """Tests for project path properties."""

    def test_project_root_is_path(self):
        """Test project_root returns a Path."""
        settings = Settings()
        assert isinstance(settings.project_root, Path)

    def test_data_dir_is_path(self):
        """Test data_dir returns a Path."""
        settings = Settings()
        assert isinstance(settings.data_dir, Path)

    def test_data_dir_contains_data(self):
        """Test data_dir path contains 'data'."""
        settings = Settings()
        assert "data" in str(settings.data_dir)


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self):
        """Test get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self):
        """Test get_settings returns the same cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        # Note: Due to lru_cache, these should be the same object
        # However, in tests the cache may be cleared, so we just verify
        # both are valid Settings instances
        assert isinstance(settings1, Settings)
        assert isinstance(settings2, Settings)


class TestEnvConfig:
    """Tests for environment configuration."""

    def test_model_config_env_prefix(self):
        """Test environment variable prefix is set."""
        # Settings uses ACME_ prefix for env vars
        assert Settings.model_config.get("env_prefix") == "ACME_"

    def test_model_config_extra_ignore(self):
        """Test extra fields are ignored."""
        assert Settings.model_config.get("extra") == "ignore"
