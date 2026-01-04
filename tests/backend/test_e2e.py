"""
End-to-end integration tests for the backend.

Tests the complete flow from API request through agent pipeline to response.
These tests use mocked LLM to ensure deterministic results.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Set mock mode
os.environ["MOCK_LLM"] = "1"


# =============================================================================
# E2E API Tests
# =============================================================================

# =============================================================================
# E2E Agent Pipeline Tests
# =============================================================================

class TestE2EAgentPipeline:
    """End-to-end tests for agent pipeline."""

    @pytest.mark.integration
    @patch("backend.agent.graph.agent_graph")
    def test_agent_handles_data_query(self, mock_graph):
        """Test agent handles data query end-to-end."""
        from backend.agent.graph import run_agent

        mock_graph.invoke.return_value = {
            "answer": "Acme Manufacturing has $500K in pipeline.",
            "sources": [{"type": "company", "id": "ACME-MFG", "label": "Acme Manufacturing"}],
            "steps": [
                {"id": "route", "label": "Route", "status": "done"},
                {"id": "data", "label": "Data", "status": "done"},
                {"id": "answer", "label": "Answer", "status": "done"},
            ],
            "raw_data": {"pipeline": []},
            "follow_up_suggestions": [],
            "mode_used": "data",
            "resolved_company_id": "ACME-MFG",
            "days": 90,
        }

        result = run_agent("What is the pipeline for Acme Manufacturing?")

        assert "answer" in result
        assert "sources" in result
        assert "meta" in result
        assert result["meta"]["mode_used"] == "data"

    @pytest.mark.integration
    @patch("backend.agent.graph.agent_graph")
    def test_agent_handles_docs_query(self, mock_graph):
        """Test agent handles docs query end-to-end."""
        from backend.agent.graph import run_agent

        mock_graph.invoke.return_value = {
            "answer": "To import contacts, use the Import wizard.",
            "sources": [{"type": "doc", "id": "doc-1", "label": "User Guide"}],
            "steps": [
                {"id": "route", "label": "Route", "status": "done"},
                {"id": "docs", "label": "Docs", "status": "done"},
                {"id": "answer", "label": "Answer", "status": "done"},
            ],
            "raw_data": {},
            "follow_up_suggestions": [],
            "mode_used": "docs",
        }

        result = run_agent("How do I import contacts?", mode="docs")

        assert result["meta"]["mode_used"] == "docs"

    @pytest.mark.integration
    @patch("backend.agent.graph.agent_graph")
    def test_agent_includes_latency_metric(self, mock_graph):
        """Test agent response includes latency metric."""
        from backend.agent.graph import run_agent

        mock_graph.invoke.return_value = {
            "answer": "Test answer",
            "sources": [],
            "steps": [],
            "raw_data": {},
            "follow_up_suggestions": [],
            "mode_used": "data",
        }

        result = run_agent("Test question")

        assert "meta" in result
        assert "latency_ms" in result["meta"]
        assert isinstance(result["meta"]["latency_ms"], int)

    @pytest.mark.integration
    @patch("backend.agent.graph.agent_graph")
    def test_agent_handles_error_gracefully(self, mock_graph):
        """Test agent handles errors gracefully."""
        from backend.agent.graph import run_agent

        mock_graph.invoke.side_effect = Exception("Test error")

        # use_cache=False to avoid hitting cached results from previous tests
        result = run_agent("Test question for e2e error handling", use_cache=False)

        # Should still return a valid response structure
        assert "answer" in result
        assert "meta" in result
        assert result["meta"]["mode_used"] == "error"


# =============================================================================
# E2E Datastore Tests
# =============================================================================

class TestE2EDatastore:
    """End-to-end tests for datastore operations."""

    def test_company_lookup_returns_data(self, datastore, sample_company_id):
        """Test company lookup returns valid data."""
        company = datastore.get_company(sample_company_id)

        assert company is not None
        assert company["company_id"] == sample_company_id
        assert "name" in company

    def test_activities_lookup_returns_data(self, datastore, sample_company_id):
        """Test activities lookup returns valid data."""
        activities = datastore.get_recent_activities(sample_company_id)

        assert isinstance(activities, list)
        # Should have some activities for the sample company

    def test_pipeline_lookup_returns_data(self, datastore, sample_company_id):
        """Test pipeline lookup returns valid data."""
        pipeline = datastore.get_pipeline_summary(sample_company_id)

        assert isinstance(pipeline, dict)  # Returns summary dict, not list

    def test_nonexistent_company_returns_none(self, datastore):
        """Test that nonexistent company returns None."""
        company = datastore.get_company("NONEXISTENT-ID")

        assert company is None


# =============================================================================
# E2E Data Explorer Tests
# =============================================================================

class TestE2EDataExplorer:
    """End-to-end tests for data explorer endpoints."""

    def test_data_companies_returns_list(self, client):
        """Test /api/data/companies returns company list."""
        response = client.get("/api/data/companies")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert isinstance(data["data"], list)

    def test_data_activities_returns_list(self, client):
        """Test /api/data/activities returns activities list."""
        response = client.get("/api/data/activities")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert isinstance(data["data"], list)

    def test_data_opportunities_returns_list(self, client):
        """Test /api/data/opportunities returns opportunities list."""
        response = client.get("/api/data/opportunities")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert isinstance(data["data"], list)

    def test_data_contacts_returns_list(self, client):
        """Test /api/data/contacts returns contacts list."""
        response = client.get("/api/data/contacts")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert isinstance(data["data"], list)


# =============================================================================
# E2E Streaming Tests
# =============================================================================

class TestE2EStreaming:
    """End-to-end tests for streaming functionality."""

    def test_stream_endpoint_returns_sse(self, client):
        """Test streaming endpoint returns SSE format."""
        with patch("backend.api.chat.stream_agent") as mock_stream:
            async def mock_generator():
                yield 'event: status\ndata: {"message": "Starting..."}\n\n'
                yield 'event: done\ndata: {"answer": "Test answer"}\n\n'

            mock_stream.return_value = mock_generator()

            response = client.post(
                "/api/chat/stream",
                json={"question": "Test question"}
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

    def test_stream_content_format(self, client):
        """Test streaming content is properly formatted."""
        with patch("backend.api.chat.stream_agent") as mock_stream:
            events = [
                'event: status\ndata: {"node": "route", "message": "Routing..."}\n\n',
                'event: step\ndata: {"id": "route", "status": "done"}\n\n',
                'event: answer_start\ndata: {}\n\n',
                'event: answer_chunk\ndata: {"chunk": "Test"}\n\n',
                'event: answer_end\ndata: {"answer": "Test"}\n\n',
                'event: done\ndata: {"answer": "Test"}\n\n',
            ]

            async def mock_generator():
                for event in events:
                    yield event

            mock_stream.return_value = mock_generator()

            response = client.post(
                "/api/chat/stream",
                json={"question": "Test"}
            )

            content = response.text

            # Verify SSE format
            assert "event: status" in content or "event: done" in content


# =============================================================================
# E2E Error Handling Tests
# =============================================================================

class TestE2EErrorHandling:
    """End-to-end tests for error handling."""

    def test_stream_invalid_json_returns_422(self, client):
        """Test that invalid JSON returns 422."""
        response = client.post(
            "/api/chat/stream",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_stream_missing_question_returns_422(self, client):
        """Test that missing question returns 422."""
        response = client.post("/api/chat/stream", json={})

        assert response.status_code == 422

    def test_stream_empty_question_returns_400(self, client):
        """Test that empty question returns 400."""
        response = client.post("/api/chat/stream", json={"question": ""})

        assert response.status_code == 400

    def test_stream_question_too_long_returns_400(self, client):
        """Test that overly long question returns 400."""
        response = client.post(
            "/api/chat/stream",
            json={"question": "x" * 2001}
        )

        assert response.status_code == 400
