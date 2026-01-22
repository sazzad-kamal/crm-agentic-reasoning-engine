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

    def test_root_redirects_to_docs(self):
        """Test that root path redirects to /docs."""
        from backend.main import app

        client = TestClient(app, follow_redirects=False)
        response = client.get("/")

        assert response.status_code in [301, 302, 307, 308]
        assert "/docs" in response.headers.get("location", "")

    def test_root_redirect_is_followed(self):
        """Test that redirect leads to docs page."""
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
