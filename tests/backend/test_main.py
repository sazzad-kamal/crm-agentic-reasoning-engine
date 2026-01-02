"""
Tests for backend/main.py.

Tests the application factory, lifespan, exception handlers, and other main.py functionality.

Run with:
    pytest tests/backend/test_main.py -v
"""

import os
import pytest
import logging
from unittest.mock import patch, MagicMock

os.environ["MOCK_LLM"] = "1"

from fastapi import FastAPI
from fastapi.testclient import TestClient


# =============================================================================
# Setup Logging Tests
# =============================================================================


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_configures_root_logger(self):
        """Test that setup_logging configures logging."""
        from backend.startup import setup_logging

        setup_logging()

        root_logger = logging.getLogger()
        assert root_logger.level != logging.NOTSET or len(root_logger.handlers) > 0

    def test_setup_logging_reduces_third_party_noise(self):
        """Test that third-party loggers are quieted."""
        from backend.startup import setup_logging

        setup_logging()

        httpx_logger = logging.getLogger("httpx")
        assert httpx_logger.level >= logging.WARNING

        httpcore_logger = logging.getLogger("httpcore")
        assert httpcore_logger.level >= logging.WARNING


# =============================================================================
# Ensure RAG Collections Tests
# =============================================================================


class TestEnsureRagCollections:
    """Tests for ensure_rag_collections_exist function."""

    @patch("qdrant_client.QdrantClient")
    @patch("backend.agent.rag_tools.ingest_docs")
    @patch("backend.agent.rag_tools.ingest_private_texts")
    def test_creates_docs_collection_when_missing(
        self, mock_ingest_private, mock_ingest_docs, mock_qdrant_class
    ):
        """Test that docs collection is created when missing."""
        from backend.startup import ensure_rag_collections_exist

        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        mock_client.collection_exists.side_effect = [False, True]
        mock_info = MagicMock()
        mock_info.points_count = 10
        mock_client.get_collection.return_value = mock_info

        ensure_rag_collections_exist()

        mock_ingest_docs.assert_called_once()

    @patch("qdrant_client.QdrantClient")
    @patch("backend.agent.rag_tools.ingest_docs")
    @patch("backend.agent.rag_tools.ingest_private_texts")
    def test_creates_private_collection_when_missing(
        self, mock_ingest_private, mock_ingest_docs, mock_qdrant_class
    ):
        """Test that private collection is created when missing."""
        from backend.startup import ensure_rag_collections_exist

        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        mock_client.collection_exists.side_effect = [True, False]
        mock_info = MagicMock()
        mock_info.points_count = 10
        mock_client.get_collection.return_value = mock_info

        ensure_rag_collections_exist()

        mock_ingest_private.assert_called_once()

    @patch("qdrant_client.QdrantClient")
    @patch("backend.agent.rag_tools.ingest_docs")
    @patch("backend.agent.rag_tools.ingest_private_texts")
    def test_ingests_when_docs_collection_empty(
        self, mock_ingest_private, mock_ingest_docs, mock_qdrant_class
    ):
        """Test that docs are ingested when collection is empty."""
        from backend.startup import ensure_rag_collections_exist

        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        mock_client.collection_exists.return_value = True

        def get_collection_side_effect(name):
            info = MagicMock()
            if name == "acme_crm_docs":
                info.points_count = 0
            else:
                info.points_count = 10
            return info

        mock_client.get_collection.side_effect = get_collection_side_effect

        ensure_rag_collections_exist()

        mock_ingest_docs.assert_called_once()

    @patch("qdrant_client.QdrantClient")
    @patch("backend.agent.rag_tools.ingest_docs")
    @patch("backend.agent.rag_tools.ingest_private_texts")
    def test_ingests_when_private_collection_empty(
        self, mock_ingest_private, mock_ingest_docs, mock_qdrant_class
    ):
        """Test that private texts are ingested when collection is empty."""
        from backend.startup import ensure_rag_collections_exist

        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        mock_client.collection_exists.return_value = True

        def get_collection_side_effect(name):
            info = MagicMock()
            if "private" in name.lower():
                info.points_count = 0
            else:
                info.points_count = 10
            return info

        mock_client.get_collection.side_effect = get_collection_side_effect

        ensure_rag_collections_exist()

        mock_ingest_private.assert_called_once()

    @patch("qdrant_client.QdrantClient")
    @patch("backend.agent.rag_tools.ingest_docs")
    @patch("backend.agent.rag_tools.ingest_private_texts")
    def test_skips_ingestion_when_collections_have_data(
        self, mock_ingest_private, mock_ingest_docs, mock_qdrant_class
    ):
        """Test that ingestion is skipped when collections have data."""
        from backend.startup import ensure_rag_collections_exist

        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        mock_client.collection_exists.return_value = True
        mock_info = MagicMock()
        mock_info.points_count = 100
        mock_client.get_collection.return_value = mock_info

        ensure_rag_collections_exist()

        mock_ingest_docs.assert_not_called()
        mock_ingest_private.assert_not_called()

    @patch("qdrant_client.QdrantClient")
    def test_raises_on_error(self, mock_qdrant_class):
        """Test that errors are propagated."""
        from backend.startup import ensure_rag_collections_exist

        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        mock_client.collection_exists.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            ensure_rag_collections_exist()

    @patch("qdrant_client.QdrantClient")
    def test_closes_client_on_success(self, mock_qdrant_class):
        """Test that Qdrant client is closed on success."""
        from backend.startup import ensure_rag_collections_exist

        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        mock_client.collection_exists.return_value = True
        mock_info = MagicMock()
        mock_info.points_count = 10
        mock_client.get_collection.return_value = mock_info

        ensure_rag_collections_exist()

        mock_client.close.assert_called()

    @patch("qdrant_client.QdrantClient")
    def test_closes_client_on_error(self, mock_qdrant_class):
        """Test that Qdrant client is closed even on error."""
        from backend.startup import ensure_rag_collections_exist

        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        mock_client.collection_exists.side_effect = Exception("Error")

        with pytest.raises(Exception):
            ensure_rag_collections_exist()

        mock_client.close.assert_called()


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
        from backend.exceptions import APIError, ValidationError

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
# Lifespan Tests
# =============================================================================


class TestLifespan:
    """Tests for lifespan context manager."""

    def test_lifespan_logs_startup(self):
        """Test that lifespan logs startup message."""
        import anyio
        from backend.startup import lifespan

        with patch("backend.startup.ensure_rag_collections_exist"):
            with patch("backend.startup.logger") as mock_logger:
                app = FastAPI()

                async def run():
                    async with lifespan(app):
                        pass

                anyio.run(run)

                assert mock_logger.info.called

    def test_lifespan_calls_ensure_rag(self):
        """Test that lifespan calls ensure_rag_collections_exist."""
        import anyio
        from backend.startup import lifespan

        with patch("backend.startup.ensure_rag_collections_exist") as mock_ensure_rag:
            with patch("backend.startup.logger"):
                app = FastAPI()

                async def run():
                    async with lifespan(app):
                        pass

                anyio.run(run)

                mock_ensure_rag.assert_called_once()

    def test_lifespan_logs_shutdown(self):
        """Test that lifespan logs shutdown message."""
        import anyio
        from backend.startup import lifespan

        with patch("backend.startup.ensure_rag_collections_exist"):
            with patch("backend.startup.logger") as mock_logger:
                app = FastAPI()

                async def run():
                    async with lifespan(app):
                        pass

                anyio.run(run)

                shutdown_logged = any(
                    "Shutting down" in str(call) for call in mock_logger.info.call_args_list
                )
                assert shutdown_logged


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

    def test_api_routes_have_cache_control(self):
        """Test that API routes have cache control headers."""
        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/health")

        assert "cache-control" in response.headers
        assert "no-store" in response.headers["cache-control"]
