"""Tests for backend configuration."""

from pathlib import Path

from backend.core.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_app_name(self):
        """Test default app name."""
        settings = Settings()
        assert settings.app_name == "Acme CRM AI Companion API"

    def test_debug_is_bool(self):
        """Test debug is a boolean."""
        settings = Settings()
        assert isinstance(settings.debug, bool)

    def test_log_level_is_string(self):
        """Test log level is a string."""
        settings = Settings()
        assert isinstance(settings.log_level, str)
        assert settings.log_level in ("DEBUG", "INFO", "WARNING", "ERROR")

    def test_log_requests_default(self):
        """Test request logging is enabled by default."""
        settings = Settings()
        assert settings.log_requests is True


class TestCorsSettings:
    """Tests for CORS configuration."""

    def test_default_cors_origins(self):
        """Test default CORS origins contains localhost."""
        settings = Settings()
        assert "http://localhost:5173" in settings.cors_origins

    def test_cors_origins_is_list(self):
        """Test CORS origins is a list."""
        settings = Settings()
        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) > 0


class TestProjectPaths:
    """Tests for project path properties."""

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


class TestEnvConfig:
    """Tests for environment configuration."""

    def test_model_config_env_prefix(self):
        """Test environment variable prefix is set."""
        assert Settings.model_config.get("env_prefix") == "ACME_"

    def test_model_config_extra_ignore(self):
        """Test extra fields are ignored."""
        assert Settings.model_config.get("extra") == "ignore"
