"""
Tests for the backend.agent module.

These tests use MOCK_LLM=1 to avoid requiring an API key.
Run with: MOCK_LLM=1 pytest backend/agent/tests/test_agent.py -v
"""

import os
import pytest

# Set mock mode before imports (also set in conftest.py)
os.environ["MOCK_LLM"] = "1"

from backend.agent.datastore import CRMDataStore, get_csv_base_path
from backend.agent.tools import (
    tool_company_lookup,
    tool_recent_activity,
    tool_recent_history,
    tool_pipeline,
    tool_upcoming_renewals,
)
from backend.agent.router import route_question
from backend.agent.orchestrator import answer_question

# datastore fixture is provided by conftest.py


# =============================================================================
# Datastore Tests
# =============================================================================

class TestDatastore:
    """Tests for CRMDataStore."""
    
    def test_csv_path_detection(self):
        """Test that CSV path is detected correctly."""
        path = get_csv_base_path()
        assert path.exists(), f"CSV path should exist: {path}"
    
    def test_resolve_company_id_exact(self, datastore):
        """Test resolving company by exact ID."""
        company_id = datastore.resolve_company_id("ACME-MFG")
        assert company_id == "ACME-MFG"
    
    def test_resolve_company_id_by_name(self, datastore):
        """Test resolving company by name."""
        company_id = datastore.resolve_company_id("Acme Manufacturing")
        assert company_id == "ACME-MFG"
    
    def test_resolve_company_id_case_insensitive(self, datastore):
        """Test resolving company name is case insensitive."""
        company_id = datastore.resolve_company_id("acme manufacturing")
        assert company_id == "ACME-MFG"
    
    def test_resolve_company_id_fuzzy(self, datastore):
        """Test fuzzy matching for company names."""
        # "Acme" should fuzzy match to "Acme Manufacturing"
        company_id = datastore.resolve_company_id("Acme Mfg")
        assert company_id is not None
    
    def test_resolve_company_id_not_found(self, datastore):
        """Test resolving non-existent company."""
        company_id = datastore.resolve_company_id("NonExistent Corp XYZ123")
        assert company_id is None
    
    def test_get_company(self, datastore):
        """Test getting company details."""
        company = datastore.get_company("ACME-MFG")
        assert company is not None
        assert company["company_id"] == "ACME-MFG"
        assert company["name"] == "Acme Manufacturing"
        assert "status" in company
        assert "plan" in company
    
    def test_get_company_not_found(self, datastore):
        """Test getting non-existent company."""
        company = datastore.get_company("NONEXISTENT")
        assert company is None
    
    def test_get_recent_activities(self, datastore):
        """Test getting recent activities."""
        activities = datastore.get_recent_activities("ACME-MFG", days=365)
        assert isinstance(activities, list)
        # Should have at least some activities
        if activities:
            assert "activity_id" in activities[0]
            assert "type" in activities[0]
    
    def test_get_recent_history(self, datastore):
        """Test getting recent history."""
        history = datastore.get_recent_history("ACME-MFG", days=365)
        assert isinstance(history, list)
        if history:
            assert "history_id" in history[0]
            assert "type" in history[0]
    
    def test_get_open_opportunities(self, datastore):
        """Test getting open opportunities."""
        opps = datastore.get_open_opportunities("ACME-MFG")
        assert isinstance(opps, list)
        if opps:
            assert "opportunity_id" in opps[0]
            assert "stage" in opps[0]
    
    def test_get_pipeline_summary(self, datastore):
        """Test getting pipeline summary."""
        summary = datastore.get_pipeline_summary("ACME-MFG")
        assert "stages" in summary
        assert "total_count" in summary
        assert "total_value" in summary
    
    def test_get_upcoming_renewals(self, datastore):
        """Test getting upcoming renewals."""
        renewals = datastore.get_upcoming_renewals(days=365)
        assert isinstance(renewals, list)
        if renewals:
            assert "company_id" in renewals[0]
            assert "renewal_date" in renewals[0]


# =============================================================================
# Tools Tests
# =============================================================================

class TestTools:
    """Tests for tool functions."""
    
    def test_tool_company_lookup_found(self):
        """Test company lookup when found."""
        result = tool_company_lookup("ACME-MFG")
        assert result.data["found"] is True
        assert result.data["company"]["company_id"] == "ACME-MFG"
        assert len(result.sources) > 0
        assert result.sources[0].type == "company"
    
    def test_tool_company_lookup_by_name(self):
        """Test company lookup by name."""
        result = tool_company_lookup("Acme Manufacturing")
        assert result.data["found"] is True
        assert result.data["company"]["company_id"] == "ACME-MFG"
    
    def test_tool_company_lookup_not_found(self):
        """Test company lookup when not found."""
        result = tool_company_lookup("Totally Unknown Company XYZ")
        assert result.data["found"] is False
        assert result.error is not None
        assert "close_matches" in result.data
    
    def test_tool_recent_activity(self):
        """Test recent activity tool."""
        result = tool_recent_activity("ACME-MFG", days=365)
        assert "activities" in result.data
        assert "count" in result.data
        assert isinstance(result.data["activities"], list)
    
    def test_tool_recent_history(self):
        """Test recent history tool."""
        result = tool_recent_history("ACME-MFG", days=365)
        assert "history" in result.data
        assert "count" in result.data
    
    def test_tool_pipeline(self):
        """Test pipeline tool."""
        result = tool_pipeline("ACME-MFG")
        assert "summary" in result.data
        assert "opportunities" in result.data
        assert "total_count" in result.data["summary"]
    
    def test_tool_upcoming_renewals(self):
        """Test upcoming renewals tool."""
        result = tool_upcoming_renewals(days=365)
        assert "renewals" in result.data
        assert "count" in result.data


# =============================================================================
# Router Tests
# =============================================================================

class TestRouter:
    """Tests for the router."""
    
    def test_route_docs_question(self):
        """Test routing a docs question."""
        result = route_question("How do I create a new opportunity?")
        assert result.mode_used in ["docs", "data+docs"]
    
    def test_route_data_question(self):
        """Test routing a data question."""
        result = route_question("What's going on with Acme Manufacturing in the last 90 days?")
        assert result.mode_used in ["data", "data+docs"]
        assert result.company_id == "ACME-MFG"
    
    def test_route_renewals_question(self):
        """Test routing a renewals question."""
        result = route_question("Which accounts have upcoming renewals?")
        assert result.mode_used in ["data", "data+docs"]
        assert result.intent == "renewals"
    
    def test_route_pipeline_question(self):
        """Test routing a pipeline question."""
        result = route_question("Show the open pipeline for Beta Tech")
        assert result.mode_used in ["data", "data+docs"]
        assert result.intent == "pipeline"
    
    def test_route_extracts_days(self):
        """Test that router extracts days from question."""
        result = route_question("What happened in the last 30 days?")
        assert result.days == 30
        
        result = route_question("What happened in the last 90 days?")
        assert result.days == 90
    
    def test_route_explicit_mode(self):
        """Test explicit mode override."""
        result = route_question("Tell me about Acme", mode="docs")
        assert result.mode_used == "docs"
        
        result = route_question("Tell me about Acme", mode="data")
        assert result.mode_used == "data"


# =============================================================================
# Agent Tests
# =============================================================================

class TestAgent:
    """Tests for the agent with MOCK_LLM."""
    
    def test_answer_question_company(self):
        """Test answering a company question."""
        result = answer_question("What's going on with Acme Manufacturing in the last 90 days?")
        
        # Check all required keys exist
        assert "answer" in result
        assert "sources" in result
        assert "steps" in result
        assert "raw_data" in result
        assert "meta" in result
        
        # Check meta
        assert "mode_used" in result["meta"]
        assert "latency_ms" in result["meta"]
        
        # Check raw_data structure
        assert "companies" in result["raw_data"]
        assert "activities" in result["raw_data"]
        assert "opportunities" in result["raw_data"]
        
        # Should have company data
        assert len(result["raw_data"]["companies"]) > 0
        assert result["raw_data"]["companies"][0]["company_id"] == "ACME-MFG"
        
        # Should have sources
        assert len(result["sources"]) > 0
    
    def test_answer_question_renewals(self):
        """Test answering a renewals question."""
        result = answer_question("Which accounts have upcoming renewals in the next 90 days?")
        
        assert "answer" in result
        assert "raw_data" in result
        assert "renewals" in result["raw_data"]
    
    def test_answer_question_pipeline(self):
        """Test answering a pipeline question."""
        result = answer_question("Show the open pipeline for Beta Tech Solutions")
        
        assert "answer" in result
        assert "raw_data" in result
        assert "opportunities" in result["raw_data"]
    
    def test_answer_question_steps(self):
        """Test that steps are returned correctly."""
        result = answer_question("What's going on with Acme Manufacturing?")
        
        steps = result["steps"]
        step_ids = [s["id"] for s in steps]
        
        # Should have router step
        assert "router" in step_ids
        
        # Should have answer step
        assert "answer" in step_ids
        
        # All steps should have status
        for step in steps:
            assert "status" in step
            assert step["status"] in ["done", "error", "skipped"]
    
    def test_answer_question_sources_non_empty(self):
        """Test that sources are returned."""
        result = answer_question("What's going on with Acme Manufacturing?")
        
        assert len(result["sources"]) > 0
        
        for source in result["sources"]:
            assert "type" in source
            assert "id" in source
            assert "label" in source
    
    def test_answer_question_company_not_found(self):
        """Test handling of unknown company."""
        result = answer_question("What's going on with NonExistent Corp XYZ?")
        
        # Should still return valid response
        assert "answer" in result
        assert "meta" in result


# =============================================================================
# Integration Tests (API)
# =============================================================================

class TestAPIIntegration:
    """Tests for the FastAPI endpoint."""
    
    def test_chat_endpoint(self):
        """Test the /api/chat endpoint."""
        from fastapi.testclient import TestClient
        from backend.main import app
        
        client = TestClient(app)
        
        response = client.post(
            "/api/chat",
            json={"question": "What's going on with Acme Manufacturing?"}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check all required response keys
        assert "answer" in data
        assert "sources" in data
        assert "steps" in data
        assert "raw_data" in data
        assert "meta" in data
    
    def test_chat_endpoint_with_mode(self):
        """Test the /api/chat endpoint with explicit mode."""
        from fastapi.testclient import TestClient
        from backend.main import app
        
        client = TestClient(app)
        
        response = client.post(
            "/api/chat",
            json={"question": "How do I create an opportunity?", "mode": "docs"}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["meta"]["mode_used"] == "docs"
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        from fastapi.testclient import TestClient
        from backend.main import app
        
        client = TestClient(app)
        
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
