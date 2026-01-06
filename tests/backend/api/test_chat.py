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

        assert response.status_code == 422

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
        from backend.agent.nodes.support.streaming import _format_sse as format_sse

        result = format_sse("status", {"message": "Testing"})

        assert result.startswith("event: status\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_format_sse_serializes_json(self):
        """Test that format_sse properly serializes JSON."""
        from backend.agent.nodes.support.streaming import _format_sse as format_sse

        result = format_sse("test", {"key": "value", "num": 42})

        # Extract JSON data
        data_line = [line for line in result.split("\n") if line.startswith("data: ")][0]
        json_str = data_line.replace("data: ", "")
        parsed = json.loads(json_str)

        assert parsed["key"] == "value"
        assert parsed["num"] == 42

    def test_stream_event_types_defined(self):
        """Test that all expected stream event types are defined."""
        from backend.agent.nodes.support.streaming import StreamEvent

        assert hasattr(StreamEvent, "STATUS")
        assert hasattr(StreamEvent, "ANSWER_START")
        assert hasattr(StreamEvent, "ANSWER_CHUNK")
        assert hasattr(StreamEvent, "ANSWER_END")
        assert hasattr(StreamEvent, "FOLLOWUP")
        assert hasattr(StreamEvent, "DONE")
        assert hasattr(StreamEvent, "ERROR")

    def test_node_messages_defined(self):
        """Test that node messages are defined for expected nodes."""
        from backend.agent.nodes.support.streaming import NODE_MESSAGES

        assert "route" in NODE_MESSAGES
        assert "fetch" in NODE_MESSAGES
        assert "answer" in NODE_MESSAGES
        assert "followup" in NODE_MESSAGES


# =============================================================================
# Starter Questions Endpoint Tests
# =============================================================================

class TestStarterQuestionsEndpoint:
    """Tests for GET /api/chat/starter-questions."""

    def test_returns_starter_questions(self, client: TestClient):
        """Should return list of starter questions."""
        response = client.get("/api/chat/starter-questions")
        assert response.status_code == 200
        questions = response.json()
        assert isinstance(questions, list)

    def test_returns_expected_count(self, client: TestClient):
        """Should return expected number of starter questions."""
        response = client.get("/api/chat/starter-questions")
        questions = response.json()
        # Should have at least 3 starter questions
        assert len(questions) >= 3

    def test_questions_are_strings(self, client: TestClient):
        """All questions should be strings."""
        response = client.get("/api/chat/starter-questions")
        questions = response.json()
        for question in questions:
            assert isinstance(question, str)
            assert len(question) > 0

    def test_contains_expected_starters(self, client: TestClient):
        """Should contain expected starter questions."""
        response = client.get("/api/chat/starter-questions")
        questions = response.json()
        # Should have at least one company-specific and one general question
        has_acme = any("Acme" in q for q in questions)
        has_beta = any("Beta" in q for q in questions)
        has_renewal = any("renewal" in q.lower() for q in questions)
        assert has_acme or has_beta or has_renewal
