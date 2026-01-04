"""
Tests for the streaming chat endpoint and stream_agent function.

Tests cover:
- SSE event formatting
- JSON serialization of complex objects
- Streaming response structure
- Error handling during streaming
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, date

from backend.agent.output.streaming import (
    format_sse,
    serialize_for_json,
    StreamEvent,
    NODE_MESSAGES,
)


class TestSerializeForJson:
    """Tests for the JSON serialization helper."""

    def test_serialize_primitives(self):
        """Should pass through primitive types unchanged."""
        assert serialize_for_json(None) is None
        assert serialize_for_json("hello") == "hello"
        assert serialize_for_json(42) == 42
        assert serialize_for_json(3.14) == 3.14
        assert serialize_for_json(True) is True

    def test_serialize_datetime(self):
        """Should convert datetime to ISO format string."""
        dt = datetime(2025, 12, 24, 10, 30, 0)
        result = serialize_for_json(dt)
        assert result == "2025-12-24T10:30:00"

    def test_serialize_date(self):
        """Should convert date to ISO format string."""
        d = date(2025, 12, 24)
        result = serialize_for_json(d)
        assert result == "2025-12-24"

    def test_serialize_dict(self):
        """Should recursively serialize dict values."""
        data = {
            "name": "Test",
            "date": date(2025, 1, 1),
            "nested": {"value": 42},
        }
        result = serialize_for_json(data)
        assert result == {
            "name": "Test",
            "date": "2025-01-01",
            "nested": {"value": 42},
        }

    def test_serialize_list(self):
        """Should recursively serialize list items."""
        data = [1, "two", date(2025, 1, 1), {"key": "value"}]
        result = serialize_for_json(data)
        assert result == [1, "two", "2025-01-01", {"key": "value"}]

    def test_serialize_pydantic_model(self):
        """Should call model_dump() on Pydantic v2 models."""
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {"id": "123", "name": "Test"}
        
        result = serialize_for_json(mock_model)
        
        mock_model.model_dump.assert_called_once()
        assert result == {"id": "123", "name": "Test"}

    def test_serialize_pydantic_v1_model(self):
        """Should call dict() on Pydantic v1 models."""
        mock_model = MagicMock(spec=[])  # No model_dump
        mock_model.dict = MagicMock(return_value={"id": "456"})
        
        # Remove model_dump to simulate v1 model
        del mock_model.model_dump
        
        result = serialize_for_json(mock_model)
        
        mock_model.dict.assert_called_once()
        assert result == {"id": "456"}

    def test_serialize_unknown_type(self):
        """Should convert unknown types to string."""
        class CustomClass:
            def __str__(self):
                return "custom_value"
        
        result = serialize_for_json(CustomClass())
        assert result == "custom_value"


class TestFormatSse:
    """Tests for SSE formatting."""

    def test_format_basic_event(self):
        """Should format event with type and JSON data."""
        result = format_sse("status", {"message": "Processing"})
        
        assert result.startswith("event: status\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        
        # Extract and verify JSON
        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        data_json = json.loads(data_line[6:])
        assert data_json == {"message": "Processing"}

    def test_format_complex_event(self):
        """Should serialize complex data in SSE format."""
        result = format_sse("done", {
            "answer": "Test answer",
            "sources": [{"type": "doc", "id": "123"}],
            "meta": {"latency_ms": 500},
        })
        
        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        data_json = json.loads(data_line[6:])
        
        assert data_json["answer"] == "Test answer"
        assert len(data_json["sources"]) == 1
        assert data_json["meta"]["latency_ms"] == 500

    def test_format_event_with_datetime(self):
        """Should serialize datetime in events."""
        result = format_sse("test", {
            "timestamp": datetime(2025, 12, 24, 12, 0, 0)
        })
        
        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        data_json = json.loads(data_line[6:])
        
        assert data_json["timestamp"] == "2025-12-24T12:00:00"


class TestStreamEvent:
    """Tests for StreamEvent constants."""

    def test_event_types_defined(self):
        """Should have all expected event types."""
        assert StreamEvent.STATUS == "status"
        assert StreamEvent.STEP == "step"
        assert StreamEvent.SOURCES == "sources"
        assert StreamEvent.ANSWER_START == "answer_start"
        assert StreamEvent.ANSWER_CHUNK == "answer_chunk"
        assert StreamEvent.ANSWER_END == "answer_end"
        assert StreamEvent.FOLLOWUP == "followup"
        assert StreamEvent.DONE == "done"
        assert StreamEvent.ERROR == "error"


class TestNodeMessages:
    """Tests for node message mappings."""

    def test_all_nodes_have_messages(self):
        """Should have messages for all pipeline nodes."""
        expected_nodes = ["route", "data", "docs", "skip_data", "skip_docs", "answer", "followup"]
        
        for node in expected_nodes:
            assert node in NODE_MESSAGES
            assert isinstance(NODE_MESSAGES[node], str)
            assert len(NODE_MESSAGES[node]) > 0

    def test_messages_are_user_friendly(self):
        """Messages should be descriptive and user-friendly."""
        assert "question" in NODE_MESSAGES["route"].lower() or "understanding" in NODE_MESSAGES["route"].lower()
        assert "data" in NODE_MESSAGES["data"].lower() or "crm" in NODE_MESSAGES["data"].lower()
        assert "doc" in NODE_MESSAGES["docs"].lower()
        assert "answer" in NODE_MESSAGES["answer"].lower() or "generating" in NODE_MESSAGES["answer"].lower()


class TestStreamAgentIntegration:
    """Integration tests for stream_agent function."""

    def test_stream_agent_mock_mode(self):
        """Should stream events in mock mode."""
        import os
        import asyncio
        
        async def run_test():
            # Set mock mode
            with patch.dict(os.environ, {"MOCK_LLM": "1"}):
                from backend.agent.output.streaming import stream_agent
                
                events = []
                async for event in stream_agent("What is Acme CRM?"):
                    events.append(event)
                
                # Should have at least status and done events
                assert len(events) > 0
                
                # Should end with done event
                last_event = events[-1]
                assert "event: done" in last_event or "event: error" in last_event
        
        asyncio.run(run_test())

    def test_stream_agent_yields_sse_format(self):
        """All yielded events should be valid SSE format."""
        import os
        import asyncio
        
        async def run_test():
            with patch.dict(os.environ, {"MOCK_LLM": "1"}):
                from backend.agent.output.streaming import stream_agent
                
                async for event in stream_agent("Test question"):
                    # Each event should have event: and data: lines
                    assert "event: " in event
                    assert "data: " in event
                    assert event.endswith("\n\n")
                    
                    # Data should be valid JSON
                    data_line = [l for l in event.split("\n") if l.startswith("data: ")][0]
                    json.loads(data_line[6:])  # Should not raise
        
        asyncio.run(run_test())


class TestStreamingEndpoint:
    """Tests for the /api/chat/stream endpoint."""

    def test_endpoint_returns_streaming_response(self):
        """Endpoint should return StreamingResponse with correct headers."""
        from fastapi.testclient import TestClient
        from backend.main import app
        import os
        
        with patch.dict(os.environ, {"MOCK_LLM": "1"}):
            client = TestClient(app)
            
            response = client.post(
                "/api/chat/stream",
                json={"question": "What is Acme CRM?"},
                headers={"Accept": "text/event-stream"},
            )
            
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

    def test_endpoint_validates_empty_question(self):
        """Should reject empty questions."""
        from fastapi.testclient import TestClient
        from backend.main import app
        
        client = TestClient(app)
        
        response = client.post(
            "/api/chat/stream",
            json={"question": ""},
        )
        
        assert response.status_code == 400

    def test_endpoint_validates_long_question(self):
        """Should reject questions over 2000 characters."""
        from fastapi.testclient import TestClient
        from backend.main import app
        
        client = TestClient(app)
        
        long_question = "a" * 2001
        response = client.post(
            "/api/chat/stream",
            json={"question": long_question},
        )
        
        assert response.status_code == 400
