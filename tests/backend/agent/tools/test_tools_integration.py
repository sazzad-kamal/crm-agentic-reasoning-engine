"""
Integration tests for agent tools.

Tests each agent tool for correctness:
- company_lookup: Does it find the right company?
- recent_activity: Does it return correct activities?
- recent_history: Does it return correct history?
- pipeline: Does it return correct opportunities?
- upcoming_renewals: Does it return correct renewals?
- search_contacts: Does it find contacts correctly?
- search_companies: Does it find companies correctly?
- pipeline_summary: Does it aggregate pipeline data?
- search_attachments: Does it find attachments?
- search_activities: Does it search activities?

Run with: pytest tests/backend/agent/tools/test_tools_integration.py -v
"""

import pytest

pytestmark = pytest.mark.integration

from backend.agent.datastore import get_datastore
from backend.agent.fetch.tools import (
    tool_company_lookup,
    tool_search_attachments,
    tool_search_companies,
    tool_search_contacts,
    tool_pipeline,
    tool_pipeline_summary,
    tool_upcoming_renewals,
    tool_analytics,
    tool_recent_activity,
    tool_recent_history,
    tool_search_activities,
)


@pytest.fixture(scope="module")
def datastore():
    """Initialize datastore once for all tests."""
    return get_datastore()


# =============================================================================
# Company Lookup Tests
# =============================================================================


class TestCompanyLookup:
    """Tests for company_lookup tool."""

    @pytest.mark.parametrize(
        "company_id_or_name,expected_id",
        [
            ("ACME-MFG", "ACME-MFG"),
            ("GREEN-ENERGY", "GREEN-ENERGY"),
            ("Acme Manufacturing", "ACME-MFG"),
            ("Beta Tech Solutions", "BETA-TECH"),
            ("Crown", "CROWN-FOODS"),  # Partial match
        ],
    )
    def test_company_lookup_found(self, datastore, company_id_or_name: str, expected_id: str):
        """Test that company lookup finds companies by ID or name."""
        result = tool_company_lookup(company_id_or_name=company_id_or_name)
        assert result.data.get("found") is True, f"Expected to find {company_id_or_name}"
        assert result.data.get("company", {}).get("company_id") == expected_id

    def test_company_lookup_not_found(self, datastore):
        """Test that company lookup returns not found for nonexistent company."""
        result = tool_company_lookup(company_id_or_name="Nonexistent Corp")
        assert result.data.get("found") is False


# =============================================================================
# Recent Activity Tests
# =============================================================================


class TestRecentActivity:
    """Tests for recent_activity tool."""

    @pytest.mark.parametrize(
        "company_id",
        ["ACME-MFG", "GREEN-ENERGY", "BETA-TECH"],
    )
    def test_recent_activity_found(self, datastore, company_id: str):
        """Test that recent activity returns results for valid companies."""
        result = tool_recent_activity(company_id=company_id, days=365)
        activities = result.data.get("activities", [])
        assert len(activities) >= 1, f"Expected at least 1 activity for {company_id}"


# =============================================================================
# Recent History Tests
# =============================================================================


class TestRecentHistory:
    """Tests for recent_history tool."""

    @pytest.mark.parametrize(
        "company_id,min_count",
        [
            ("ACME-MFG", 3),
            ("GREEN-ENERGY", 3),
            ("HARBOR-LOGISTICS", 2),
        ],
    )
    def test_recent_history_found(self, datastore, company_id: str, min_count: int):
        """Test that recent history returns minimum expected results."""
        result = tool_recent_history(company_id=company_id, days=365)
        history = result.data.get("history", [])
        assert len(history) >= min_count, f"Expected at least {min_count} history for {company_id}"


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestPipeline:
    """Tests for pipeline tool."""

    @pytest.mark.parametrize(
        "company_id,expected_found,min_count",
        [
            ("ACME-MFG", True, 1),
            ("GREEN-ENERGY", False, 0),  # Churned - no open opps
            ("FUSION-RETAIL", True, 1),
        ],
    )
    def test_pipeline(self, datastore, company_id: str, expected_found: bool, min_count: int):
        """Test pipeline tool returns correct opportunities."""
        result = tool_pipeline(company_id=company_id)
        opportunities = result.data.get("opportunities", [])
        actual_found = len(opportunities) > 0
        assert actual_found == expected_found, f"Expected found={expected_found} for {company_id}"
        assert len(opportunities) >= min_count


# =============================================================================
# Upcoming Renewals Tests
# =============================================================================


class TestUpcomingRenewals:
    """Tests for upcoming_renewals tool."""

    def test_renewals_365d(self, datastore):
        """Test renewals within 365 days returns results."""
        result = tool_upcoming_renewals(days=365)
        renewals = result.data.get("renewals", [])
        assert len(renewals) >= 1, "Expected at least 1 renewal within 365 days"

    def test_renewals_90d(self, datastore):
        """Test renewals within 90 days (may or may not have results)."""
        result = tool_upcoming_renewals(days=90)
        # Just ensure it doesn't error - may or may not have renewals
        assert "renewals" in result.data


# =============================================================================
# Search Contacts Tests
# =============================================================================


class TestSearchContacts:
    """Tests for search_contacts tool."""

    def test_search_contacts_by_company(self, datastore):
        """Test searching contacts by company."""
        result = tool_search_contacts(company_id="ACME-MFG")
        contacts = result.data.get("contacts", [])
        assert len(contacts) >= 1

    @pytest.mark.parametrize("role", ["Decision Maker", "Champion"])
    def test_search_contacts_by_role(self, datastore, role: str):
        """Test searching contacts by role."""
        result = tool_search_contacts(role=role)
        contacts = result.data.get("contacts", [])
        assert len(contacts) >= 1, f"Expected at least 1 {role}"

    def test_search_contacts_all(self, datastore):
        """Test searching all contacts."""
        result = tool_search_contacts()
        contacts = result.data.get("contacts", [])
        assert len(contacts) >= 5


# =============================================================================
# Search Companies Tests
# =============================================================================


class TestSearchCompanies:
    """Tests for search_companies tool."""

    @pytest.mark.parametrize("segment", ["Mid-market", "SMB"])
    def test_search_companies_by_segment(self, datastore, segment: str):
        """Test searching companies by segment."""
        result = tool_search_companies(segment=segment)
        companies = result.data.get("companies", [])
        assert len(companies) >= 1, f"Expected at least 1 {segment} company"

    def test_search_companies_by_industry(self, datastore):
        """Test searching companies by industry."""
        result = tool_search_companies(industry="Software")
        companies = result.data.get("companies", [])
        assert len(companies) >= 1

    def test_search_companies_all(self, datastore):
        """Test searching all companies."""
        result = tool_search_companies()
        companies = result.data.get("companies", [])
        assert len(companies) >= 5


# =============================================================================
# Pipeline Summary Tests
# =============================================================================


class TestPipelineSummary:
    """Tests for pipeline_summary tool."""

    def test_pipeline_summary(self, datastore):
        """Test pipeline summary aggregation."""
        result = tool_pipeline_summary()
        assert result.data.get("total_count", 0) >= 1
        assert "total_value" in result.data
        assert result.data.get("total_value", 0) >= 0


# =============================================================================
# Search Attachments Tests
# =============================================================================


class TestSearchAttachments:
    """Tests for search_attachments tool."""

    def test_search_attachments_all(self, datastore):
        """Test searching all attachments."""
        result = tool_search_attachments()
        attachments = result.data.get("attachments", [])
        assert len(attachments) >= 1

    def test_search_attachments_query(self, datastore):
        """Test searching attachments by query."""
        result = tool_search_attachments(query="proposal")
        attachments = result.data.get("attachments", [])
        assert len(attachments) >= 1

    def test_search_attachments_by_company(self, datastore):
        """Test searching attachments by company (may or may not exist)."""
        result = tool_search_attachments(company_id="ACME-MFG")
        # Just ensure it doesn't error
        assert "attachments" in result.data


# =============================================================================
# Search Activities Tests
# =============================================================================


class TestSearchActivities:
    """Tests for search_activities tool."""

    def test_search_activities_all(self, datastore):
        """Test searching all activities."""
        result = tool_search_activities(days=365)
        activities = result.data.get("activities", [])
        assert len(activities) >= 1

    @pytest.mark.parametrize("activity_type", ["Call", "Meeting"])
    def test_search_activities_by_type(self, datastore, activity_type: str):
        """Test searching activities by type (may or may not exist)."""
        result = tool_search_activities(activity_type=activity_type, days=365)
        # Just ensure it doesn't error
        assert "activities" in result.data


# =============================================================================
# Analytics Tests
# =============================================================================


class TestAnalytics:
    """Tests for tool_analytics."""

    def test_contact_breakdown(self, datastore):
        """Test contact breakdown by role."""
        result = tool_analytics(metric="contact_breakdown", group_by="role")
        assert "breakdown" in result.data
        assert len(result.data["breakdown"]) >= 1

    def test_contact_breakdown_by_company(self, datastore):
        """Test contact breakdown filtered by company."""
        result = tool_analytics(metric="contact_breakdown", company_id="ACME-MFG")
        assert "breakdown" in result.data

    def test_activity_breakdown(self, datastore):
        """Test activity breakdown by type."""
        result = tool_analytics(metric="activity_breakdown", group_by="type", days=365)
        assert "breakdown" in result.data
        assert len(result.data["breakdown"]) >= 1

    def test_activity_count(self, datastore):
        """Test activity count with no filter."""
        result = tool_analytics(metric="activity_count", days=365)
        assert "count" in result.data
        assert result.data["count"] >= 0

    def test_activity_count_by_type(self, datastore):
        """Test activity count filtered by type."""
        result = tool_analytics(metric="activity_count", activity_type="Call", days=365)
        assert "count" in result.data

    def test_accounts_by_group(self, datastore):
        """Test accounts by group aggregation."""
        result = tool_analytics(metric="accounts_by_group")
        assert "breakdown" in result.data

    def test_pipeline_by_group(self, datastore):
        """Test pipeline value by group."""
        result = tool_analytics(metric="pipeline_by_group")
        assert "breakdown" in result.data
