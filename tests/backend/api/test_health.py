"""Tests for API routes."""

import pytest
from unittest.mock import patch
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient

from backend.api.chat import router as chat_router
from backend.api.health import router as health_router, HealthResponse, SystemInfo
from backend.api.data import router as data_router, DataResponse, load_csv_data, load_jsonl_data

# Combined API router (same as main.py)
router = APIRouter(prefix="/api")
router.include_router(chat_router, tags=["chat"])
router.include_router(health_router, tags=["health"])
router.include_router(data_router, tags=["data"])


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def app():
    """Create test FastAPI app."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


# =============================================================================
# Model Tests
# =============================================================================

class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response_creation(self):
        """Test HealthResponse with required fields."""
        response = HealthResponse(status="ok")
        assert response.status == "ok"
        assert response.services == {}

    def test_health_response_with_services(self):
        """Test HealthResponse with services dict."""
        services = {"api": "healthy", "data": "healthy"}
        response = HealthResponse(status="ok", services=services)
        assert response.services["api"] == "healthy"


class TestSystemInfo:
    """Tests for SystemInfo model."""

    def test_system_info_creation(self):
        """Test SystemInfo with all fields."""
        info = SystemInfo(
            app_name="Test App",
            debug=False,
            cors_origins=["http://localhost:3000"],
        )
        assert info.app_name == "Test App"
        assert info.debug is False
        assert len(info.cors_origins) == 1


class TestDataResponse:
    """Tests for DataResponse model."""

    def test_data_response_creation(self):
        """Test DataResponse with all fields."""
        response = DataResponse(
            data=[{"id": 1, "name": "Test"}],
            total=1,
            columns=["id", "name"],
        )
        assert len(response.data) == 1
        assert response.total == 1
        assert response.columns == ["id", "name"]

    def test_data_response_empty(self):
        """Test DataResponse with empty data."""
        response = DataResponse(data=[], total=0, columns=[])
        assert response.data == []
        assert response.total == 0


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestLoadCsvData:
    """Tests for load_csv_data helper."""

    def test_nonexistent_file_returns_empty(self, tmp_path):
        """Test nonexistent file returns empty list."""
        path = tmp_path / "nonexistent.csv"
        data, columns = load_csv_data(path)
        assert data == []
        assert columns == []

    def test_loads_csv_file(self, tmp_path):
        """Test loading a valid CSV file."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("id,name\n1,Alice\n2,Bob\n")

        data, columns = load_csv_data(csv_path)

        assert len(data) == 2
        assert columns == ["id", "name"]
        assert data[0]["name"] == "Alice"


class TestLoadJsonlData:
    """Tests for load_jsonl_data helper."""

    def test_nonexistent_file_returns_empty(self, tmp_path):
        """Test nonexistent file returns empty list."""
        path = tmp_path / "nonexistent.jsonl"
        data, columns = load_jsonl_data(path)
        assert data == []
        assert columns == []

    def test_loads_jsonl_file(self, tmp_path):
        """Test loading a valid JSONL file."""
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text('{"id": 1, "name": "Alice"}\n{"id": 2, "name": "Bob"}\n')

        data, columns = load_jsonl_data(jsonl_path)

        assert len(data) == 2
        assert "id" in columns
        assert "name" in columns


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Test health endpoint returns status field."""
        response = client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_returns_services(self, client):
        """Test health endpoint returns services dict."""
        response = client.get("/api/health")
        data = response.json()
        assert "services" in data
        assert isinstance(data["services"], dict)


# =============================================================================
# Info Endpoint Tests
# =============================================================================

class TestInfoEndpoint:
    """Tests for /api/info endpoint."""

    def test_info_returns_200(self, client):
        """Test info endpoint returns 200."""
        response = client.get("/api/info")
        assert response.status_code == 200

    def test_info_returns_app_name(self, client):
        """Test info endpoint returns app_name."""
        response = client.get("/api/info")
        data = response.json()
        assert "app_name" in data
        assert isinstance(data["app_name"], str)

    def test_info_returns_debug(self, client):
        """Test info endpoint returns debug flag."""
        response = client.get("/api/info")
        data = response.json()
        assert "debug" in data
        assert isinstance(data["debug"], bool)

    def test_info_returns_cors_origins(self, client):
        """Test info endpoint returns cors_origins."""
        response = client.get("/api/info")
        data = response.json()
        assert "cors_origins" in data
        assert isinstance(data["cors_origins"], list)


# =============================================================================
# Chat Endpoint Tests
# =============================================================================

class TestChatEndpoint:
    """Tests for /api/chat endpoint."""

    def test_chat_requires_question(self, client):
        """Test chat endpoint requires question field."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422  # Validation error

    def test_chat_validates_empty_question(self, client):
        """Test chat validates empty question."""
        response = client.post("/api/chat", json={"question": ""})
        # Should fail validation (400 or 422)
        assert response.status_code in [400, 422]

    def test_chat_validates_long_question(self, client):
        """Test chat validates question length."""
        long_question = "x" * 2001  # Over 2000 char limit
        response = client.post("/api/chat", json={"question": long_question})
        assert response.status_code in [400, 422]

    @patch("backend.api.chat.answer_question")
    def test_chat_calls_answer_question(self, mock_answer, client):
        """Test chat calls answer_question with params."""
        mock_answer.return_value = {
            "answer": "Test answer",
            "sources": [],
            "steps": [],
            "raw_data": {
                "companies": [],
                "activities": [],
                "opportunities": [],
                "history": [],
                "renewals": [],
                "pipeline_summary": None,
            },
            "meta": {
                "mode_used": "data",
                "latency_ms": 100,
            },
            "follow_up_suggestions": [],
        }

        response = client.post(
            "/api/chat",
            json={"question": "What is Acme?", "mode": "auto"},
        )

        assert response.status_code == 200
        mock_answer.assert_called_once()

    @patch("backend.api.chat.answer_question")
    def test_chat_response_structure(self, mock_answer, client):
        """Test chat response has expected structure."""
        mock_answer.return_value = {
            "answer": "Acme is a company.",
            "sources": [{"type": "company", "id": "C001", "label": "Acme"}],
            "steps": [{"id": "route", "label": "Routing", "status": "done"}],
            "raw_data": {
                "companies": [{"company_id": "C001"}],
                "activities": [],
                "opportunities": [],
                "history": [],
                "renewals": [],
                "pipeline_summary": None,
            },
            "meta": {
                "mode_used": "data",
                "latency_ms": 150,
                "company_id": "C001",
            },
            "follow_up_suggestions": ["Tell me more"],
        }

        response = client.post("/api/chat", json={"question": "What is Acme?"})
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "steps" in data
        assert "raw_data" in data
        assert "meta" in data


# =============================================================================
# Stream Endpoint Tests
# =============================================================================

class TestChatStreamEndpoint:
    """Tests for /api/chat/stream endpoint."""

    def test_stream_validates_empty_question(self, client):
        """Test stream validates empty question."""
        response = client.post("/api/chat/stream", json={"question": ""})
        assert response.status_code in [400, 422]

    def test_stream_validates_long_question(self, client):
        """Test stream validates question length."""
        long_question = "x" * 2001
        response = client.post("/api/chat/stream", json={"question": long_question})
        assert response.status_code in [400, 422]

    @patch("backend.api.chat.stream_agent")
    def test_stream_returns_streaming_response(self, mock_stream, client):
        """Test stream returns SSE response."""
        async def mock_generator():
            yield "data: {}\n\n"

        mock_stream.return_value = mock_generator()

        response = client.post(
            "/api/chat/stream",
            json={"question": "Test question"},
        )

        # Should return event-stream content type
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
