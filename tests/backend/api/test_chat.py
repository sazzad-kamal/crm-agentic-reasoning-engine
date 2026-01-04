"""
Tests for backend.api.chat module.

Tests the chat and streaming endpoints.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Set mock mode before imports
os.environ["MOCK_LLM"] = "1"

from backend.main import app


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Streaming Endpoint Tests
# =============================================================================

class TestChatStreamEndpoint:
    """Tests for POST /api/chat/stream."""

    def test_stream_returns_sse_content_type(self, client):
        """Test that stream endpoint returns SSE content type."""
        with patch("backend.api.chat.stream_agent") as mock_stream:
            async def mock_generator():
                yield 'event: status\ndata: {"message": "Starting..."}\n\n'
                yield 'event: done\ndata: {"answer": "Test"}\n\n'

            mock_stream.return_value = mock_generator()

            response = client.post(
                "/api/chat/stream",
                json={"question": "Test question"}
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

    def test_stream_validates_empty_question(self, client):
        """Test that stream endpoint rejects empty questions."""
        response = client.post(
            "/api/chat/stream",
            json={"question": ""}
        )

        assert response.status_code == 400

    def test_stream_validates_long_question(self, client):
        """Test that stream endpoint rejects overly long questions."""
        response = client.post(
            "/api/chat/stream",
            json={"question": "x" * 2001}
        )

        assert response.status_code == 400

    def test_stream_includes_cache_headers(self, client):
        """Test that stream response includes proper cache headers."""
        with patch("backend.api.chat.stream_agent") as mock_stream:
            async def mock_generator():
                yield 'event: done\ndata: {"answer": "Test"}\n\n'

            mock_stream.return_value = mock_generator()

            response = client.post(
                "/api/chat/stream",
                json={"question": "Test question"}
            )

            # Check that cache-control prevents caching (may have additional directives)
            cache_control = response.headers.get("cache-control", "")
            assert "no-cache" in cache_control


# =============================================================================
# Streaming Module Tests
# =============================================================================

class TestStreamingModule:
    """Tests for the streaming module functions."""

    def test_format_sse_returns_valid_format(self):
        """Test that format_sse returns valid SSE format."""
        from backend.agent.output.streaming import format_sse

        result = format_sse("status", {"message": "Testing"})

        assert result.startswith("event: status\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_format_sse_serializes_json(self):
        """Test that format_sse properly serializes JSON."""
        from backend.agent.output.streaming import format_sse

        result = format_sse("test", {"key": "value", "num": 42})

        # Extract JSON data
        data_line = [line for line in result.split("\n") if line.startswith("data: ")][0]
        json_str = data_line.replace("data: ", "")
        parsed = json.loads(json_str)

        assert parsed["key"] == "value"
        assert parsed["num"] == 42

    def test_serialize_for_json_handles_primitives(self):
        """Test that serialize_for_json handles primitive types."""
        from backend.agent.output.streaming import serialize_for_json

        assert serialize_for_json("test") == "test"
        assert serialize_for_json(42) == 42
        assert serialize_for_json(3.14) == 3.14
        assert serialize_for_json(True) is True
        assert serialize_for_json(None) is None

    def test_serialize_for_json_handles_lists(self):
        """Test that serialize_for_json handles lists."""
        from backend.agent.output.streaming import serialize_for_json

        result = serialize_for_json([1, "two", 3.0])
        assert result == [1, "two", 3.0]

    def test_serialize_for_json_handles_dicts(self):
        """Test that serialize_for_json handles dicts."""
        from backend.agent.output.streaming import serialize_for_json

        result = serialize_for_json({"a": 1, "b": "two"})
        assert result == {"a": 1, "b": "two"}

    def test_serialize_for_json_handles_datetime(self):
        """Test that serialize_for_json handles datetime objects."""
        from backend.agent.output.streaming import serialize_for_json
        from datetime import datetime

        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = serialize_for_json(dt)

        assert result == "2024-01-15T10:30:00"

    def test_serialize_for_json_handles_pydantic_models(self):
        """Test that serialize_for_json handles Pydantic models."""
        from backend.agent.output.streaming import serialize_for_json
        from backend.agent.core.schemas import Source

        source = Source(type="company", id="ACME", label="Acme Corp")
        result = serialize_for_json(source)

        assert result["type"] == "company"
        assert result["id"] == "ACME"
        assert result["label"] == "Acme Corp"

    def test_stream_event_types_defined(self):
        """Test that all expected stream event types are defined."""
        from backend.agent.output.streaming import StreamEvent

        assert hasattr(StreamEvent, "STATUS")
        assert hasattr(StreamEvent, "STEP")
        assert hasattr(StreamEvent, "SOURCES")
        assert hasattr(StreamEvent, "ANSWER_START")
        assert hasattr(StreamEvent, "ANSWER_CHUNK")
        assert hasattr(StreamEvent, "ANSWER_END")
        assert hasattr(StreamEvent, "FOLLOWUP")
        assert hasattr(StreamEvent, "DONE")
        assert hasattr(StreamEvent, "ERROR")

    def test_node_messages_defined(self):
        """Test that node messages are defined for expected nodes."""
        from backend.agent.output.streaming import NODE_MESSAGES

        assert "route" in NODE_MESSAGES
        assert "data" in NODE_MESSAGES
        assert "docs" in NODE_MESSAGES
        assert "answer" in NODE_MESSAGES
        assert "followup" in NODE_MESSAGES
