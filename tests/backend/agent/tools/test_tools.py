"""Tests for agent tools."""

import pytest
from unittest.mock import MagicMock, patch

from backend.agent.fetch.tools import (
    make_sources,
    tool_company_lookup,
    tool_recent_activity,
    tool_recent_history,
    tool_pipeline,
    tool_upcoming_renewals,
    ToolResult,
)
from backend.agent.core import Source


class TestMakeSources:
    """Tests for make_sources helper."""

    def test_returns_source_for_non_empty_list(self):
        """Test returns source when data is non-empty list."""
        sources = make_sources([1, 2, 3], "company", "C001", "Acme Corp")
        assert len(sources) == 1
        assert sources[0].type == "company"
        assert sources[0].id == "C001"
        assert sources[0].label == "Acme Corp"

    def test_returns_source_for_non_empty_dict(self):
        """Test returns source when data is non-empty dict."""
        sources = make_sources({"key": "value"}, "doc", "D001", "Document")
        assert len(sources) == 1

    def test_returns_empty_for_empty_list(self):
        """Test returns empty for empty list."""
        sources = make_sources([], "company", "C001", "Acme")
        assert sources == []

    def test_returns_empty_for_none(self):
        """Test returns empty for None."""
        sources = make_sources(None, "company", "C001", "Acme")
        assert sources == []

    def test_returns_empty_for_empty_dict(self):
        """Test returns empty for empty dict."""
        sources = make_sources({}, "company", "C001", "Acme")
        assert sources == []


class TestToolCompanyLookup:
    """Tests for tool_company_lookup."""

    def test_returns_tool_result(self):
        """Test returns ToolResult instance."""
        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = "C001"
        mock_ds.get_company.return_value = {"company_id": "C001", "name": "Acme Corp"}
        mock_ds.get_contacts_for_company.return_value = []
        
        result = tool_company_lookup("Acme", datastore=mock_ds)
        assert isinstance(result, ToolResult)

    def test_found_company_has_data(self):
        """Test found company includes company data."""
        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = "C001"
        mock_ds.get_company.return_value = {"company_id": "C001", "name": "Acme Corp"}
        mock_ds.get_contacts_for_company.return_value = [{"name": "John"}]
        
        result = tool_company_lookup("Acme", datastore=mock_ds)
        
        assert result.data["found"] is True
        assert result.data["company"]["name"] == "Acme Corp"
        assert len(result.data["contacts"]) == 1

    def test_found_company_has_sources(self):
        """Test found company includes sources."""
        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = "C001"
        mock_ds.get_company.return_value = {"company_id": "C001", "name": "Acme Corp"}
        mock_ds.get_contacts_for_company.return_value = []
        
        result = tool_company_lookup("C001", datastore=mock_ds)
        
        assert len(result.sources) == 1
        assert result.sources[0].type == "company"
        assert result.sources[0].id == "C001"

    def test_not_found_returns_error(self):
        """Test not found returns error."""
        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = None
        mock_ds.get_company_name_matches.return_value = []
        
        result = tool_company_lookup("Unknown", datastore=mock_ds)
        
        assert result.data["found"] is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_not_found_includes_close_matches(self):
        """Test not found includes close matches."""
        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = None
        mock_ds.get_company_name_matches.return_value = [
            {"company_id": "C001", "name": "Acme Corp"},
            {"company_id": "C002", "name": "Acme Inc"},
        ]
        
        result = tool_company_lookup("Acmee", datastore=mock_ds)
        
        assert len(result.data["close_matches"]) == 2


class TestToolRecentActivity:
    """Tests for tool_recent_activity."""

    def test_returns_tool_result(self):
        """Test returns ToolResult instance."""
        mock_ds = MagicMock()
        mock_ds.get_recent_activities.return_value = []
        mock_ds.get_company.return_value = {"name": "Acme"}
        
        result = tool_recent_activity("C001", datastore=mock_ds)
        assert isinstance(result, ToolResult)

    def test_includes_activities_in_data(self):
        """Test includes activities in result data."""
        mock_ds = MagicMock()
        mock_ds.get_recent_activities.return_value = [
            {"id": "A001", "type": "call"},
            {"id": "A002", "type": "email"},
        ]
        mock_ds.get_company.return_value = {"name": "Acme"}
        
        result = tool_recent_activity("C001", days=30, limit=10, datastore=mock_ds)
        
        assert result.data["company_id"] == "C001"
        assert result.data["days"] == 30
        assert result.data["count"] == 2
        assert len(result.data["activities"]) == 2

    def test_calls_datastore_with_params(self):
        """Test passes parameters to datastore."""
        mock_ds = MagicMock()
        mock_ds.get_recent_activities.return_value = []
        mock_ds.get_company.return_value = {"name": "Acme"}
        
        tool_recent_activity("C001", days=60, limit=15, datastore=mock_ds)
        
        mock_ds.get_recent_activities.assert_called_once_with("C001", days=60, limit=15)

    def test_includes_sources_when_has_activities(self):
        """Test includes sources when activities exist."""
        mock_ds = MagicMock()
        mock_ds.get_recent_activities.return_value = [{"id": "A001"}]
        mock_ds.get_company.return_value = {"name": "Acme"}
        
        result = tool_recent_activity("C001", datastore=mock_ds)
        
        assert len(result.sources) == 1
        assert result.sources[0].type == "activities"


class TestToolRecentHistory:
    """Tests for tool_recent_history."""

    def test_returns_tool_result(self):
        """Test returns ToolResult instance."""
        mock_ds = MagicMock()
        mock_ds.get_recent_history.return_value = []
        mock_ds.get_company.return_value = {"name": "Acme"}
        
        result = tool_recent_history("C001", datastore=mock_ds)
        assert isinstance(result, ToolResult)

    def test_includes_history_in_data(self):
        """Test includes history entries in result data."""
        mock_ds = MagicMock()
        mock_ds.get_recent_history.return_value = [
            {"id": "H001", "type": "note"},
        ]
        mock_ds.get_company.return_value = {"name": "Acme"}
        
        result = tool_recent_history("C001", datastore=mock_ds)
        
        assert result.data["company_id"] == "C001"
        assert result.data["count"] == 1
        assert len(result.data["history"]) == 1

    def test_calls_datastore_with_params(self):
        """Test passes parameters to datastore."""
        mock_ds = MagicMock()
        mock_ds.get_recent_history.return_value = []
        mock_ds.get_company.return_value = {"name": "Acme"}
        
        tool_recent_history("C001", days=45, limit=25, datastore=mock_ds)
        
        mock_ds.get_recent_history.assert_called_once_with("C001", days=45, limit=25)


class TestToolPipeline:
    """Tests for tool_pipeline."""

    def test_returns_tool_result(self):
        """Test returns ToolResult instance."""
        mock_ds = MagicMock()
        mock_ds.get_pipeline_summary.return_value = {}
        mock_ds.get_open_opportunities.return_value = []
        mock_ds.get_company.return_value = {"name": "Acme"}
        
        result = tool_pipeline("C001", datastore=mock_ds)
        assert isinstance(result, ToolResult)

    def test_includes_summary_and_opportunities(self):
        """Test includes both summary and opportunities."""
        mock_ds = MagicMock()
        mock_ds.get_pipeline_summary.return_value = {
            "total_count": 3,
            "total_value": 100000,
        }
        mock_ds.get_open_opportunities.return_value = [
            {"id": "O001", "value": 50000},
        ]
        mock_ds.get_company.return_value = {"name": "Acme"}
        
        result = tool_pipeline("C001", datastore=mock_ds)
        
        assert result.data["company_id"] == "C001"
        assert result.data["summary"]["total_count"] == 3
        assert len(result.data["opportunities"]) == 1

    def test_sources_when_has_pipeline(self):
        """Test includes sources when pipeline exists."""
        mock_ds = MagicMock()
        mock_ds.get_pipeline_summary.return_value = {"total_count": 1}
        mock_ds.get_open_opportunities.return_value = [{"id": "O001"}]
        mock_ds.get_company.return_value = {"name": "Acme"}
        
        result = tool_pipeline("C001", datastore=mock_ds)
        
        assert len(result.sources) == 1
        assert result.sources[0].type == "opportunities"


class TestToolUpcomingRenewals:
    """Tests for tool_upcoming_renewals."""

    def test_returns_tool_result(self):
        """Test returns ToolResult instance."""
        mock_ds = MagicMock()
        mock_ds.get_upcoming_renewals.return_value = []
        
        result = tool_upcoming_renewals(datastore=mock_ds)
        assert isinstance(result, ToolResult)

    def test_includes_renewals_data(self):
        """Test includes renewals in result data."""
        mock_ds = MagicMock()
        mock_ds.get_upcoming_renewals.return_value = [
            {"company_id": "C001", "renewal_date": "2025-03-01"},
            {"company_id": "C002", "renewal_date": "2025-04-01"},
        ]
        
        result = tool_upcoming_renewals(days=60, limit=10, datastore=mock_ds)
        
        assert result.data["days"] == 60
        assert result.data["count"] == 2
        assert len(result.data["renewals"]) == 2

    def test_calls_datastore_with_params(self):
        """Test passes parameters to datastore."""
        mock_ds = MagicMock()
        mock_ds.get_upcoming_renewals.return_value = []

        tool_upcoming_renewals(days=120, limit=50, datastore=mock_ds)

        mock_ds.get_upcoming_renewals.assert_called_once_with(days=120, limit=50, owner=None)

    def test_default_params(self):
        """Test uses default parameters."""
        mock_ds = MagicMock()
        mock_ds.get_upcoming_renewals.return_value = []

        tool_upcoming_renewals(datastore=mock_ds)

        mock_ds.get_upcoming_renewals.assert_called_once_with(days=90, limit=20, owner=None)
