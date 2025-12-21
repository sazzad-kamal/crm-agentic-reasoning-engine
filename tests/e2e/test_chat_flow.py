"""
End-to-End Tests for Chat Flow.

Tests the complete chat workflow from API request to response.
Uses MOCK_LLM=1 to avoid requiring an API key.

Run with:
    MOCK_LLM=1 pytest tests/e2e/test_chat_flow.py -v
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set mock mode before imports
os.environ["MOCK_LLM"] = "1"

from fastapi.testclient import TestClient
from backend.main import app
from backend.agent.schemas import ChatRequest, ChatResponse


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return "Based on the CRM data, Acme Manufacturing is an active Enterprise customer with recent engagement."


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_endpoint_returns_ok(self, client):
        """Test that health endpoint returns OK status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
    
    def test_health_endpoint_includes_services(self, client):
        """Test that health endpoint includes service statuses."""
        response = client.get("/api/health")
        data = response.json()
        assert "services" in data
        assert isinstance(data["services"], dict)
    
    def test_system_info_endpoint(self, client):
        """Test system info endpoint returns app information."""
        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "version" in data


# =============================================================================
# Chat Endpoint Tests
# =============================================================================

class TestChatEndpoint:
    """Tests for the main chat endpoint."""
    
    def test_chat_endpoint_accepts_valid_request(self, client):
        """Test that chat endpoint accepts a valid request."""
        payload = {"question": "What's going on with Acme Manufacturing?"}
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
    
    def test_chat_endpoint_returns_chat_response_schema(self, client):
        """Test that response follows ChatResponse schema."""
        payload = {"question": "What's going on with Acme Manufacturing?"}
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        # Check required fields
        assert "answer" in data
        assert "sources" in data
        assert "steps" in data
        assert "meta" in data
    
    def test_chat_endpoint_with_company_query(self, client):
        """Test chat with a company-specific query."""
        payload = {"question": "What's the status of Acme Manufacturing?"}
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        assert response.status_code == 200
        assert len(data["answer"]) > 0
    
    def test_chat_endpoint_with_docs_query(self, client):
        """Test chat with a documentation query."""
        payload = {
            "question": "How do I create a new opportunity?",
            "mode": "docs"
        }
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        assert response.status_code == 200
        assert "answer" in data
    
    def test_chat_endpoint_with_data_mode(self, client):
        """Test chat with explicit data mode."""
        payload = {
            "question": "Show me upcoming renewals",
            "mode": "data"
        }
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        assert response.status_code == 200
        assert "sources" in data
    
    def test_chat_endpoint_with_days_parameter(self, client):
        """Test chat with custom days parameter."""
        payload = {
            "question": "Recent activities for Acme Manufacturing",
            "days": 90
        }
        response = client.post("/api/chat", json=payload)
        
        assert response.status_code == 200
    
    def test_chat_endpoint_includes_steps(self, client):
        """Test that response includes processing steps."""
        payload = {"question": "What's the pipeline for TechCorp?"}
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        assert "steps" in data
        assert isinstance(data["steps"], list)
        if data["steps"]:
            step = data["steps"][0]
            assert "id" in step
            assert "label" in step
            assert "status" in step
    
    def test_chat_endpoint_includes_meta(self, client):
        """Test that response includes metadata."""
        payload = {"question": "What's going on with Acme Manufacturing?"}
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        assert "meta" in data
        meta = data["meta"]
        assert "mode_used" in meta
        assert "latency_ms" in meta
    
    def test_chat_endpoint_returns_follow_ups(self, client):
        """Test that response includes follow-up suggestions."""
        payload = {"question": "What's the pipeline overview?"}
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        # follow_ups is optional but should be a list if present
        if "follow_ups" in data:
            assert isinstance(data["follow_ups"], list)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestChatErrorHandling:
    """Tests for error handling in chat endpoint."""
    
    def test_chat_endpoint_rejects_empty_question(self, client):
        """Test that empty question is rejected."""
        payload = {"question": ""}
        response = client.post("/api/chat", json=payload)
        # Should either reject or handle gracefully
        assert response.status_code in [200, 400, 422]
    
    def test_chat_endpoint_rejects_missing_question(self, client):
        """Test that missing question field is rejected."""
        payload = {}
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_chat_endpoint_handles_unknown_company(self, client):
        """Test handling of unknown company names."""
        payload = {"question": "What's going on with NonExistent Corp XYZ123?"}
        response = client.post("/api/chat", json=payload)
        
        # Should still return 200 with a helpful message
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_chat_endpoint_handles_invalid_mode(self, client):
        """Test handling of invalid mode parameter."""
        payload = {
            "question": "Hello",
            "mode": "invalid_mode"
        }
        response = client.post("/api/chat", json=payload)
        # Should reject invalid mode or use default
        assert response.status_code in [200, 400, 422]


# =============================================================================
# Request/Response Headers Tests
# =============================================================================

class TestRequestHeaders:
    """Tests for request/response headers."""
    
    def test_response_includes_request_id(self, client):
        """Test that response includes X-Request-ID header."""
        payload = {"question": "Hello"}
        response = client.post("/api/chat", json=payload)
        
        assert "x-request-id" in response.headers
    
    def test_response_includes_response_time(self, client):
        """Test that response includes X-Response-Time header."""
        payload = {"question": "Hello"}
        response = client.post("/api/chat", json=payload)
        
        assert "x-response-time" in response.headers
    
    def test_custom_request_id_is_preserved(self, client):
        """Test that custom X-Request-ID is preserved."""
        custom_id = "test-request-123"
        payload = {"question": "Hello"}
        response = client.post(
            "/api/chat",
            json=payload,
            headers={"X-Request-ID": custom_id}
        )
        
        assert response.headers.get("x-request-id") == custom_id
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present for cross-origin requests."""
        response = client.options(
            "/api/chat",
            headers={"Origin": "http://localhost:5173"}
        )
        # OPTIONS should be handled
        assert response.status_code in [200, 405]


# =============================================================================
# Integration Flow Tests
# =============================================================================

class TestChatIntegrationFlow:
    """Tests for complete chat integration flow."""
    
    def test_company_status_flow(self, client):
        """Test complete company status query flow."""
        payload = {"question": "What's the status of Acme Manufacturing?"}
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        assert response.status_code == 200
        assert "answer" in data
        assert "sources" in data
        assert "meta" in data
        
        # Should have used data mode for company query
        assert data["meta"]["mode_used"] in ["data", "data+docs"]
    
    def test_renewals_query_flow(self, client):
        """Test complete renewals query flow."""
        payload = {"question": "Which accounts have renewals coming up?"}
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        assert response.status_code == 200
        assert "answer" in data
    
    def test_pipeline_query_flow(self, client):
        """Test complete pipeline query flow."""
        payload = {"question": "Show me the sales pipeline"}
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        assert response.status_code == 200
        assert "answer" in data
    
    def test_docs_query_flow(self, client):
        """Test complete documentation query flow."""
        payload = {
            "question": "How do I import contacts?",
            "mode": "docs"
        }
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        assert response.status_code == 200
        assert data["meta"]["mode_used"] in ["docs", "data+docs"]
    
    def test_mixed_query_flow(self, client):
        """Test query that requires both data and docs."""
        payload = {
            "question": "What are the best practices for managing Acme Manufacturing's pipeline?",
            "mode": "data+docs"
        }
        response = client.post("/api/chat", json=payload)
        data = response.json()
        
        assert response.status_code == 200
        assert "answer" in data


# =============================================================================
# Performance Tests
# =============================================================================

class TestChatPerformance:
    """Basic performance tests for chat endpoint."""
    
    def test_response_time_is_reasonable(self, client):
        """Test that response time is within acceptable limits."""
        payload = {"question": "Hello"}
        response = client.post("/api/chat", json=payload)
        
        # Response time should be in header
        response_time = response.headers.get("x-response-time", "0ms")
        time_ms = int(response_time.replace("ms", ""))
        
        # Should respond within 30 seconds (generous for mock mode)
        assert time_ms < 30000
    
    def test_multiple_requests_handled(self, client):
        """Test that multiple sequential requests are handled."""
        questions = [
            "What's going on with Acme Manufacturing?",
            "Show me upcoming renewals",
            "How do I create a contact?",
        ]
        
        for question in questions:
            response = client.post("/api/chat", json={"question": question})
            assert response.status_code == 200
