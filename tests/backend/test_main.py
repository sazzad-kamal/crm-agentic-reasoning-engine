"""
Tests for backend/main.py.

Tests the application factory, exception handlers, and other main.py functionality.

Run with:
    pytest tests/backend/test_main.py -v
"""

import os

os.environ["MOCK_LLM"] = "1"

from fastapi import FastAPI
from fastapi.testclient import TestClient

# =============================================================================
# Create App Tests
# =============================================================================


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_create_app_returns_fastapi_instance(self):
        """Test that create_app returns a FastAPI app."""
        from backend.main import create_app

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_create_app_sets_title(self):
        """Test that app has correct title."""
        from backend.main import create_app

        app = create_app()
        assert app.title is not None
        assert len(app.title) > 0

    def test_create_app_includes_routes(self):
        """Test that app includes API routes."""
        from backend.main import create_app

        app = create_app()
        routes = [r.path for r in app.routes]

        assert any("/api" in r for r in routes)

    def test_create_app_has_docs_url(self):
        """Test that app has docs endpoint."""
        from backend.main import create_app

        app = create_app()
        assert app.docs_url == "/docs"

    def test_create_app_has_openapi_url(self):
        """Test that app has OpenAPI endpoint."""
        from backend.main import create_app

        app = create_app()
        assert app.openapi_url == "/openapi.json"


# =============================================================================
# Root Redirect Tests
# =============================================================================


class TestRootRedirect:
    """Tests for root endpoint redirect."""

    def test_root_returns_200(self):
        """Test that root path returns 200 (SPA or docs redirect)."""
        from backend.main import app

        client = TestClient(app, follow_redirects=True)
        response = client.get("/")

        assert response.status_code == 200


# =============================================================================
# App Instance Tests
# =============================================================================


class TestAppInstance:
    """Tests for the module-level app instance."""

    def test_app_is_fastapi_instance(self):
        """Test that app is a FastAPI instance."""
        from backend.main import app

        assert isinstance(app, FastAPI)


# =============================================================================
# Middleware Configuration Tests
# =============================================================================


class TestMiddlewareConfiguration:
    """Tests for middleware configuration in create_app."""

    def test_app_adds_request_id_header(self):
        """Test that app adds X-Request-ID header."""
        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/data/companies")

        assert "x-request-id" in response.headers

    def test_app_adds_response_time_header(self):
        """Test that app adds X-Response-Time header."""
        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/data/companies")

        assert "x-response-time" in response.headers


# =============================================================================
# Basic Auth Middleware Tests
# =============================================================================


class TestBasicAuth:
    """Tests for HTTP Basic Auth middleware."""

    def test_no_auth_required_when_env_vars_unset(self, monkeypatch):
        """Without AUTH_USER/AUTH_PASS, all requests pass through."""
        monkeypatch.setattr("backend.main.AUTH_USER", "")
        monkeypatch.setattr("backend.main.AUTH_PASS", "")
        from backend.main import create_app

        client = TestClient(create_app())
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_exempt_from_auth(self, monkeypatch):
        """Health endpoint is always accessible, even with auth enabled."""
        monkeypatch.setattr("backend.main.AUTH_USER", "admin")
        monkeypatch.setattr("backend.main.AUTH_PASS", "secret")
        from backend.main import create_app

        client = TestClient(create_app())
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_auth_required_returns_401(self, monkeypatch):
        """Requests without credentials return 401."""
        monkeypatch.setattr("backend.main.AUTH_USER", "admin")
        monkeypatch.setattr("backend.main.AUTH_PASS", "secret")
        from backend.main import create_app

        client = TestClient(create_app())
        response = client.get("/api/data/companies")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_valid_credentials_pass(self, monkeypatch):
        """Correct credentials allow access."""
        monkeypatch.setattr("backend.main.AUTH_USER", "admin")
        monkeypatch.setattr("backend.main.AUTH_PASS", "secret")
        from backend.main import create_app

        client = TestClient(create_app())
        response = client.get("/api/data/companies", auth=("admin", "secret"))
        assert response.status_code == 200

    def test_invalid_credentials_rejected(self, monkeypatch):
        """Wrong credentials return 401."""
        monkeypatch.setattr("backend.main.AUTH_USER", "admin")
        monkeypatch.setattr("backend.main.AUTH_PASS", "secret")
        from backend.main import create_app

        client = TestClient(create_app())
        response = client.get("/api/data/companies", auth=("admin", "wrong"))
        assert response.status_code == 401

    def test_invalid_auth_format_rejected(self, monkeypatch):
        """Non-Basic auth scheme returns 401."""
        monkeypatch.setattr("backend.main.AUTH_USER", "admin")
        monkeypatch.setattr("backend.main.AUTH_PASS", "secret")
        from backend.main import create_app

        client = TestClient(create_app())
        response = client.get(
            "/api/data/companies",
            headers={"Authorization": "Bearer some-token"},
        )
        assert response.status_code == 401
