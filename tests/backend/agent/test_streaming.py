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

from backend.agent.streaming import (
    _format_sse as format_sse,
    StreamEvent,
    NODE_MESSAGES,
)


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
        assert StreamEvent.ANSWER_START == "answer_start"
        assert StreamEvent.ANSWER_CHUNK == "answer_chunk"
        assert StreamEvent.ANSWER_END == "answer_end"
        assert StreamEvent.FOLLOWUP == "followup"
        assert StreamEvent.DONE == "done"
        assert StreamEvent.ERROR == "error"


class TestNodeMessages:
    """Tests for node message mappings."""

    def test_all_nodes_have_messages(self):
        """Should have messages for all pipeline nodes (including 3 parallel fetch nodes)."""
        # With parallel fetch, we have: route, fetch_crm, fetch_docs, fetch_account, answer, followup
        expected_nodes = ["route", "fetch_crm", "fetch_docs", "fetch_account", "answer", "followup"]

        for node in expected_nodes:
            assert node in NODE_MESSAGES
            assert isinstance(NODE_MESSAGES[node], str)
            assert len(NODE_MESSAGES[node]) > 0

    def test_messages_are_user_friendly(self):
        """Messages should be descriptive and user-friendly."""
        assert "question" in NODE_MESSAGES["route"].lower() or "understanding" in NODE_MESSAGES["route"].lower()
        # Check parallel fetch nodes have appropriate messages
        assert "crm" in NODE_MESSAGES["fetch_crm"].lower() or "data" in NODE_MESSAGES["fetch_crm"].lower()
        assert "doc" in NODE_MESSAGES["fetch_docs"].lower() or "search" in NODE_MESSAGES["fetch_docs"].lower()
        assert "account" in NODE_MESSAGES["fetch_account"].lower() or "context" in NODE_MESSAGES["fetch_account"].lower()
        assert "answer" in NODE_MESSAGES["answer"].lower() or "generating" in NODE_MESSAGES["answer"].lower()


class TestStreamAnswerChain:
    """Tests for the stream_answer_chain function."""

    def test_stream_answer_chain_mock_mode(self):
        """Should yield complete answer in mock mode."""
        import os
        import asyncio

        async def run_test():
            with patch.dict(os.environ, {"MOCK_LLM": "1"}):
                from backend.agent.answer.llm import stream_answer_chain

                tokens = []
                async for token in stream_answer_chain(
                    question="What is Acme's status?",
                    conversation_history_section="",
                    company_section="",
                    activities_section="",
                    history_section="",
                    pipeline_section="",
                    renewals_section="",
                    docs_section="",
                ):
                    tokens.append(token)

                # Should yield at least one token (the mock response)
                assert len(tokens) >= 1
                # Combined should form a complete answer
                full_answer = "".join(tokens)
                assert len(full_answer) > 0

        asyncio.run(run_test())

    def test_stream_answer_chain_yields_strings(self):
        """All yielded tokens should be strings."""
        import os
        import asyncio

        async def run_test():
            with patch.dict(os.environ, {"MOCK_LLM": "1"}):
                from backend.agent.answer.llm import stream_answer_chain

                async for token in stream_answer_chain(
                    question="Test",
                    conversation_history_section="",
                    company_section="",
                    activities_section="",
                    history_section="",
                    pipeline_section="",
                    renewals_section="",
                    docs_section="",
                ):
                    assert isinstance(token, str)

        asyncio.run(run_test())


class TestStreamAgentIntegration:
    """Integration tests for stream_agent function."""

    def test_stream_agent_mock_mode(self):
        """Should stream events in mock mode."""
        import os
        import asyncio
        
        async def run_test():
            # Set mock mode
            with patch.dict(os.environ, {"MOCK_LLM": "1"}):
                from backend.agent.streaming import stream_agent

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
                from backend.agent.streaming import stream_agent
                
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

        assert response.status_code == 422
