"""
Tests for backend/main.py.

Tests the application factory, exception handlers, and other main.py functionality.

Run with:
    pytest tests/backend/test_main.py -v
"""

import os
import pytest

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
# Exception Handler Tests
# =============================================================================


class TestExceptionHandlers:
    """Tests for exception handlers defined in create_app."""

    @pytest.fixture
    def test_app(self):
        """Create test app with exception handlers."""
        from backend.main import create_app
        from backend.core.exceptions import APIError, ValidationError

        app = create_app()

        @app.get("/test/api-error")
        async def raise_api_error():
            raise APIError(status_code=400, message="Test API error")

        @app.get("/test/validation-error")
        async def raise_validation_error():
            raise ValidationError(message="Test validation error", field="test_field")

        @app.get("/test/general-error")
        async def raise_general_error():
            raise RuntimeError("Unexpected error")

        return app

    @pytest.fixture
    def test_client(self, test_app):
        """Create test client."""
        return TestClient(test_app, raise_server_exceptions=False)

    def test_api_error_handler_returns_json(self, test_client):
        """Test that APIError handler returns JSON response."""
        response = test_client.get("/test/api-error")
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"] is True

    def test_api_error_handler_includes_message(self, test_client):
        """Test that APIError handler includes error message."""
        response = test_client.get("/test/api-error")
        data = response.json()
        assert "message" in data
        assert "Test API error" in data["message"]

    def test_api_error_handler_includes_status_code(self, test_client):
        """Test that APIError handler includes status code."""
        response = test_client.get("/test/api-error")
        data = response.json()
        assert "status_code" in data
        assert data["status_code"] == 400

    def test_validation_error_handled(self, test_client):
        """Test that ValidationError is handled correctly."""
        response = test_client.get("/test/validation-error")
        assert response.status_code == 400
        data = response.json()
        assert data["error"] is True

    def test_general_exception_handler_returns_500(self, test_client):
        """Test that general exceptions return 500."""
        response = test_client.get("/test/general-error")
        assert response.status_code == 500

    def test_general_exception_returns_json(self, test_client):
        """Test that general exception handler returns JSON."""
        response = test_client.get("/test/general-error")
        data = response.json()
        assert "error" in data
        assert data["error"] is True


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

    def test_app_has_health_endpoint(self):
        """Test that app has health endpoint."""
        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_app_has_info_endpoint(self):
        """Test that app has info endpoint."""
        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/info")
        assert response.status_code == 200


# =============================================================================
# Middleware Configuration Tests
# =============================================================================


class TestMiddlewareConfiguration:
    """Tests for middleware configuration in create_app."""

    def test_app_adds_request_id_header(self):
        """Test that app adds X-Request-ID header."""
        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/health")

        assert "x-request-id" in response.headers

    def test_app_adds_response_time_header(self):
        """Test that app adds X-Response-Time header."""
        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/health")

        assert "x-response-time" in response.headers
