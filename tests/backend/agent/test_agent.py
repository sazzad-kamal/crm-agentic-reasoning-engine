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
from backend.agent.handlers import (
    tool_company_lookup,
    tool_recent_activity,
    tool_recent_history,
    tool_pipeline,
    tool_upcoming_renewals,
    tool_deals_at_risk,
    tool_forecast,
    tool_pipeline_by_owner,
)
from backend.agent.llm.router import route_question
from backend.agent.nodes.graph import run_agent

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

    def test_tool_deals_at_risk(self):
        """Test deals at risk tool."""
        result = tool_deals_at_risk()
        assert "deals" in result.data
        assert "count" in result.data
        assert isinstance(result.data["deals"], list)
        # Verify deals returned have days_in_stage (risk indicator)
        if result.data["count"] > 0:
            assert "days_in_stage" in result.data["deals"][0]

    def test_tool_deals_at_risk_with_owner(self):
        """Test deals at risk with owner filter."""
        result = tool_deals_at_risk(owner="jsmith")
        assert "deals" in result.data
        assert "owner_filter" in result.data
        assert result.data["owner_filter"] == "jsmith"

    def test_tool_forecast(self):
        """Test pipeline forecast tool."""
        result = tool_forecast()
        assert "total_pipeline" in result.data
        assert "total_weighted" in result.data
        assert "by_stage" in result.data
        # Verify weighted value is less than or equal to total (due to probability weighting)
        assert result.data["total_weighted"] <= result.data["total_pipeline"]

    def test_tool_forecast_with_owner(self):
        """Test pipeline forecast with owner filter."""
        result = tool_forecast(owner="jsmith")
        assert "total_pipeline" in result.data
        assert "owner_filter" in result.data
        assert result.data["owner_filter"] == "jsmith"

    def test_tool_pipeline_by_owner(self):
        """Test pipeline grouped by owner."""
        result = tool_pipeline_by_owner()
        assert "total_count" in result.data
        assert "total_value" in result.data
        assert "breakdown" in result.data
        assert isinstance(result.data["breakdown"], list)

    def test_tool_pipeline_by_owner_filtered(self):
        """Test pipeline filtered to specific owner."""
        result = tool_pipeline_by_owner(owner="jsmith")
        assert "total_count" in result.data
        assert "owner_filter" in result.data
        assert result.data["owner_filter"] == "jsmith"


# =============================================================================
# Router Tests (Mock Mode)
# In mock mode, router returns default routing (data+docs, general intent)
# Real routing behavior is tested via e2e_eval with actual API calls
# =============================================================================

class TestRouter:
    """Tests for the router in mock mode."""

    def test_route_returns_default_mode_in_mock(self):
        """Test that mock mode returns data+docs."""
        result = route_question("How do I create a new opportunity?")
        assert result.mode_used == "data+docs"

    def test_route_returns_default_intent_in_mock(self):
        """Test that mock mode detects intent from keywords."""
        result = route_question("What's going on with Acme Manufacturing?")
        # Mock detects company_status from "acme" keyword
        assert result.intent == "company_status"

    def test_route_returns_default_days_in_mock(self):
        """Test that mock mode returns default 30 days."""
        result = route_question("What happened in the last 90 days?")
        assert result.days == 30  # Mock mode ignores timeframe in question

    def test_route_explicit_mode_overrides(self):
        """Test explicit mode override works in mock mode."""
        result = route_question("Tell me about Acme", mode="docs")
        assert result.mode_used == "docs"

        result = route_question("Tell me about Acme", mode="data")
        assert result.mode_used == "data"

    def test_route_detects_owner_from_starter(self):
        """Test that owner detection works in mock mode."""
        result = route_question("How's my pipeline?")
        assert result.owner == "jsmith"

        result = route_question("Any renewals at risk?")
        assert result.owner == "amartin"


# =============================================================================
# Agent Tests
# =============================================================================

class TestAgent:
    """Tests for the agent with MOCK_LLM."""
    
    def test_run_agent_company(self):
        """Test answering a company question.

        Note: In mock mode, the router returns default values (general intent).
        Real company resolution is tested via e2e_eval with actual API calls.
        """
        result = run_agent("What's going on with Acme Manufacturing in the last 90 days?")

        # Check all required keys exist
        assert "answer" in result
        assert "raw_data" in result
        assert "meta" in result

        # Check meta
        assert "mode_used" in result["meta"]
        assert "latency_ms" in result["meta"]

        # Check raw_data structure exists (content depends on routing)
        assert "companies" in result["raw_data"]
        assert "activities" in result["raw_data"]
        assert "opportunities" in result["raw_data"]

        # Note: In mock mode, company resolution doesn't happen
        # Full routing behavior is tested in e2e_eval with real API calls
    
    def test_run_agent_renewals(self):
        """Test answering a renewals question."""
        result = run_agent("Which accounts have upcoming renewals in the next 90 days?")
        
        assert "answer" in result
        assert "raw_data" in result
        assert "renewals" in result["raw_data"]
    
    def test_run_agent_pipeline(self):
        """Test answering a pipeline question."""
        result = run_agent("Show the open pipeline for Beta Tech Solutions")
        
        assert "answer" in result
        assert "raw_data" in result
        assert "opportunities" in result["raw_data"]
    
    def test_run_agent_company_not_found(self):
        """Test handling of unknown company."""
        result = run_agent("What's going on with NonExistent Corp XYZ?")
        
        # Should still return valid response
        assert "answer" in result
        assert "meta" in result


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
