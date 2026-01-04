"""
End-to-End Tests for Chat Flow (API-level).

Tests the complete chat workflow from API request to response
without mocking internal agent functions. Uses MOCK_LLM=1 to avoid
requiring an API key while still testing full integration.

Run with:
    MOCK_LLM=1 pytest tests/backend/api/test_chat_e2e.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient

# Set mock mode before imports
os.environ["MOCK_LLM"] = "1"

from backend.main import app


@pytest.fixture
def client():
    """Create a test client for E2E testing."""
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Streaming Endpoint Tests
# =============================================================================

class TestStreamingEndpoint:
    """Tests for the streaming chat endpoint."""
    
    def test_streaming_endpoint_accepts_valid_request(self, client):
        """Test that streaming endpoint accepts a valid request."""
        payload = {"question": "What's going on with Acme Manufacturing?"}
        response = client.post("/api/chat/stream", json=payload)
        assert response.status_code == 200
    
    def test_streaming_endpoint_returns_event_stream(self, client):
        """Test that streaming endpoint returns correct content type."""
        payload = {"question": "What are the renewals?"}
        response = client.post("/api/chat/stream", json=payload)
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
    
    def test_streaming_response_has_cache_headers(self, client):
        """Test that streaming response has no-cache headers."""
        payload = {"question": "Show me the pipeline"}
        response = client.post("/api/chat/stream", json=payload)
        
        cache_control = response.headers.get("cache-control", "")
        assert "no-cache" in cache_control
    
    def test_streaming_events_are_valid_sse(self, client):
        """Test that streaming events follow SSE format."""
        payload = {"question": "What's going on with TechCorp?"}
        response = client.post("/api/chat/stream", json=payload)
        
        content = response.text
        
        # Should contain event: and data: lines
        assert "event:" in content
        assert "data:" in content
        
        # Each event should end with double newline
        events = content.split("\n\n")
        assert len(events) > 0
    
    def test_streaming_includes_status_events(self, client):
        """Test that streaming includes status update events."""
        payload = {"question": "How do I create a contact?"}
        response = client.post("/api/chat/stream", json=payload)
        
        content = response.text
        
        # Should have status events
        assert "event: status" in content
    
    def test_streaming_includes_step_events(self, client):
        """Test that streaming includes step completion events."""
        payload = {"question": "What renewals are coming up?"}
        response = client.post("/api/chat/stream", json=payload)
        
        content = response.text
        
        # Should have step events
        assert "event: step" in content
    
    def test_streaming_ends_with_done_event(self, client):
        """Test that streaming ends with done event."""
        payload = {"question": "What is Acme CRM?"}
        response = client.post("/api/chat/stream", json=payload)
        
        content = response.text
        
        # Should have done event at the end
        assert "event: done" in content
    
    def test_streaming_done_event_has_complete_response(self, client):
        """Test that done event contains complete response."""
        import json
        
        payload = {"question": "Show me enterprise accounts"}
        response = client.post("/api/chat/stream", json=payload)
        
        content = response.text
        
        # Find done event and parse its data
        lines = content.split("\n")
        done_data = None
        for i, line in enumerate(lines):
            if line == "event: done":
                # Next line should be data
                data_line = lines[i + 1]
                if data_line.startswith("data: "):
                    done_data = json.loads(data_line[6:])
                    break
        
        assert done_data is not None
        assert "answer" in done_data
        assert "sources" in done_data
        assert "steps" in done_data
        assert "meta" in done_data
    
    def test_streaming_rejects_empty_question(self, client):
        """Test that empty question is rejected."""
        payload = {"question": ""}
        response = client.post("/api/chat/stream", json=payload)
        
        assert response.status_code == 400
    
    def test_streaming_rejects_long_question(self, client):
        """Test that very long question is rejected."""
        payload = {"question": "a" * 2001}  # Over 2000 chars
        response = client.post("/api/chat/stream", json=payload)
        
        assert response.status_code == 400
    
    def test_streaming_with_different_modes(self, client):
        """Test streaming works with different modes."""
        modes = ["auto", "data", "docs", "data+docs"]
        
        for mode in modes:
            payload = {
                "question": "Tell me about opportunities",
                "mode": mode
            }
            response = client.post("/api/chat/stream", json=payload)
            assert response.status_code == 200
            assert "event: done" in response.text

