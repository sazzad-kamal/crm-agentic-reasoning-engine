"""
Tests for backend/agent/intent_handlers.py - intent dispatch and handlers.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from backend.agent.fetch.handlers import (
    IntentContext,
    IntentResult,
    dispatch_intent,
    INTENT_HANDLERS,
)
from backend.agent.fetch.handlers.common import empty_raw_data, safe_extend
from backend.agent.fetch.handlers.pipeline import (
    handle_pipeline_summary,
    handle_renewals,
    handle_deals_at_risk,
    handle_forecast,
)
from backend.agent.fetch.handlers.company import (
    handle_company_status,
    handle_company_search,
    handle_contacts,
    handle_attachments,
)
from backend.agent.fetch.handlers.activity import (
    handle_activities,
    handle_fallback,
)
from backend.agent.core.schemas import Source


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_tool_result():
    """Create a mock tool result."""
    result = MagicMock()
    result.data = {}
    result.sources = []
    return result


@pytest.fixture
def basic_context():
    """Create a basic IntentContext."""
    return IntentContext(
        question="test question",
        resolved_company_id=None,
        days=90,
        router_result=None,
    )


# =============================================================================
# IntentContext Tests
# =============================================================================

class TestIntentContext:
    """Tests for IntentContext dataclass."""

    def test_creates_with_required_fields(self):
        """Creates with required fields."""
        ctx = IntentContext(
            question="What is the pipeline?",
            resolved_company_id="ACME-001",
            days=30,
        )
        assert ctx.question == "What is the pipeline?"
        assert ctx.resolved_company_id == "ACME-001"
        assert ctx.days == 30
        assert ctx.router_result is None

    def test_creates_with_optional_router_result(self):
        """Creates with optional router_result."""
        router = MagicMock()
        ctx = IntentContext(
            question="test",
            resolved_company_id=None,
            days=90,
            router_result=router,
        )
        assert ctx.router_result is router


# =============================================================================
# IntentResult Tests
# =============================================================================

class TestIntentResult:
    """Tests for IntentResult dataclass."""

    def test_creates_with_defaults(self):
        """Creates with default empty values."""
        result = IntentResult()
        assert result.raw_data == {}
        assert result.sources == []
        assert result.company_data is None
        assert result.resolved_company_id is None

    def test_creates_with_data(self):
        """Creates with provided data."""
        sources = [Source(id="src1", type="company", label="Acme Corp")]
        result = IntentResult(
            raw_data={"companies": []},
            sources=sources,
            company_data={"name": "Acme"},
            resolved_company_id="ACME-001",
        )
        assert result.raw_data["companies"] == []
        assert len(result.sources) == 1
        assert result.company_data["name"] == "Acme"
        assert result.resolved_company_id == "ACME-001"


# =============================================================================
# _empty_raw_data Tests
# =============================================================================

class TestEmptyRawData:
    """Tests for _empty_raw_data helper."""

    def test_returns_empty_structure(self):
        """Returns correct empty structure."""
        data = empty_raw_data()
        assert data["companies"] == []
        assert data["contacts"] == []
        assert data["activities"] == []
        assert data["opportunities"] == []
        assert data["history"] == []
        assert data["renewals"] == []
        assert data["attachments"] == []
        assert data["pipeline_summary"] is None
        assert data["analytics"] is None


# =============================================================================
# handle_pipeline_summary Tests
# =============================================================================

class TestHandlePipelineSummary:
    """Tests for handle_pipeline_summary handler."""

    @patch('backend.agent.fetch.handlers.pipeline.tool_pipeline_summary')
    def test_fetches_pipeline_summary(self, mock_tool, basic_context):
        """Fetches pipeline summary data."""
        mock_tool.return_value.data = {
            "total_count": 10,
            "total_value": 500000,
            "by_stage": [{"stage": "Proposal", "count": 5}],
            "top_opportunities": [{"name": "Deal 1"}],
        }
        mock_tool.return_value.sources = [Source(id="pipeline", type="pipeline", label="Pipeline Summary")]

        result = handle_pipeline_summary(basic_context)

        assert result.pipeline_data["total_count"] == 10
        assert result.raw_data["pipeline_summary"]["total_count"] == 10
        assert len(result.sources) == 1


# =============================================================================
# handle_renewals Tests
# =============================================================================

class TestHandleRenewals:
    """Tests for handle_renewals handler."""

    @patch('backend.agent.fetch.handlers.pipeline.tool_upcoming_renewals')
    def test_fetches_renewals(self, mock_tool, basic_context):
        """Fetches renewal data."""
        mock_tool.return_value.data = {
            "renewals": [{"name": "Company 1", "renewal_date": "2025-03-01"}],
        }
        mock_tool.return_value.sources = []

        result = handle_renewals(basic_context)

        assert result.renewals_data is not None
        assert len(result.raw_data["renewals"]) == 1

    @patch('backend.agent.fetch.handlers.pipeline.lookup_company')
    @patch('backend.agent.fetch.handlers.pipeline.tool_upcoming_renewals')
    def test_includes_company_when_resolved(self, mock_renewals, mock_lookup):
        """Includes company data when company_id is resolved."""
        # lookup_company modifies result and returns bool
        def side_effect(result, company_id):
            result.company_data = {"found": True, "company": {"company_id": company_id}}
            result.resolved_company_id = company_id
            return True
        mock_lookup.side_effect = side_effect
        mock_renewals.return_value.data = {"renewals": []}
        mock_renewals.return_value.sources = []

        ctx = IntentContext(
            question="renewals for acme",
            resolved_company_id="ACME-001",
            days=90,
        )
        result = handle_renewals(ctx)

        # lookup_company was called with the resolved company ID
        mock_lookup.assert_called_once()
        assert mock_lookup.call_args[0][1] == "ACME-001"


# =============================================================================
# handle_contacts Tests
# =============================================================================

class TestHandleContacts:
    """Tests for handle_contacts handler."""

    @patch('backend.agent.fetch.handlers.company.tool_search_contacts')
    def test_fetches_contacts(self, mock_tool, basic_context):
        """Fetches contact data."""
        mock_tool.return_value.data = {
            "contacts": [{"name": "John Doe", "role": "CEO"}],
        }
        mock_tool.return_value.sources = []

        result = handle_contacts(basic_context)

        assert result.contacts_data is not None
        assert len(result.raw_data["contacts"]) == 1

    @patch('backend.agent.fetch.handlers.company.tool_search_contacts')
    def test_extracts_role_from_question(self, mock_tool):
        """Extracts and uses role from question."""
        mock_tool.return_value.data = {"contacts": []}
        mock_tool.return_value.sources = []

        ctx = IntentContext(
            question="who is the decision maker",
            resolved_company_id=None,
            days=90,
        )
        handle_contacts(ctx)

        mock_tool.assert_called_once()
        call_kwargs = mock_tool.call_args[1]
        assert call_kwargs["role"] == "Decision Maker"


# =============================================================================
# handle_company_search Tests
# =============================================================================

class TestHandleCompanySearch:
    """Tests for handle_company_search handler."""

    @patch('backend.agent.fetch.handlers.company.tool_search_companies')
    def test_searches_companies(self, mock_tool, basic_context):
        """Searches companies."""
        mock_tool.return_value.data = {
            "companies": [{"name": "Tech Corp", "segment": "Enterprise"}],
        }
        mock_tool.return_value.sources = []

        result = handle_company_search(basic_context)

        assert result.company_data is not None
        assert len(result.raw_data["companies"]) == 1

    @patch('backend.agent.fetch.handlers.company.tool_search_companies')
    def test_extracts_criteria_from_question(self, mock_tool):
        """Extracts segment and industry from question."""
        mock_tool.return_value.data = {"companies": []}
        mock_tool.return_value.sources = []

        ctx = IntentContext(
            question="show enterprise software companies",
            resolved_company_id=None,
            days=90,
        )
        handle_company_search(ctx)

        call_kwargs = mock_tool.call_args[1]
        assert call_kwargs["segment"] == "Enterprise"
        assert call_kwargs["industry"] == "Software"


# =============================================================================
# handle_attachments Tests
# =============================================================================

class TestHandleAttachments:
    """Tests for handle_attachments handler."""

    @patch('backend.agent.fetch.handlers.company.tool_search_attachments')
    def test_searches_attachments(self, mock_tool, basic_context):
        """Searches attachments."""
        mock_tool.return_value.data = {
            "attachments": [{"name": "Proposal.pdf"}],
        }
        mock_tool.return_value.sources = []

        result = handle_attachments(basic_context)

        assert result.attachments_data is not None
        assert len(result.raw_data["attachments"]) == 1

    @patch('backend.agent.fetch.handlers.company.tool_search_attachments')
    def test_extracts_query_from_question(self, mock_tool):
        """Extracts search query from question."""
        mock_tool.return_value.data = {"attachments": []}
        mock_tool.return_value.sources = []

        ctx = IntentContext(
            question="find the proposal document",
            resolved_company_id=None,
            days=90,
        )
        handle_attachments(ctx)

        call_kwargs = mock_tool.call_args[1]
        assert "proposal" in call_kwargs["query"]


# =============================================================================
# handle_activities Tests
# =============================================================================

class TestHandleActivities:
    """Tests for handle_activities handler."""

    @patch('backend.agent.fetch.handlers.activity.tool_search_activities')
    def test_searches_activities(self, mock_tool, basic_context):
        """Searches activities."""
        mock_tool.return_value.data = {
            "activities": [{"type": "Call", "subject": "Check-in"}],
        }
        mock_tool.return_value.sources = []

        result = handle_activities(basic_context)

        assert result.activities_data is not None
        assert len(result.raw_data["activities"]) == 1

    @patch('backend.agent.fetch.handlers.activity.tool_search_activities')
    def test_extracts_activity_type(self, mock_tool):
        """Extracts activity type from question."""
        mock_tool.return_value.data = {"activities": []}
        mock_tool.return_value.sources = []

        ctx = IntentContext(
            question="show recent calls",
            resolved_company_id=None,
            days=90,
        )
        handle_activities(ctx)

        call_kwargs = mock_tool.call_args[1]
        assert call_kwargs["activity_type"] == "Call"


# =============================================================================
# handle_company_status Tests
# =============================================================================

class TestHandleCompanyStatus:
    """Tests for handle_company_status handler."""

    @patch('backend.agent.fetch.handlers.pipeline.tool_pipeline')
    @patch('backend.agent.fetch.handlers.activity.tool_recent_history')
    @patch('backend.agent.fetch.handlers.activity.tool_recent_activity')
    @patch('backend.agent.fetch.handlers.company.lookup_company')
    def test_fetches_full_company_data(
        self, mock_lookup, mock_activity, mock_history, mock_pipeline
    ):
        """Fetches all company data when found."""
        # lookup_company modifies result and returns bool
        def side_effect(result, company_id):
            result.company_data = {"found": True, "company": {"company_id": "ACME-001", "name": "Acme Corp"}}
            result.resolved_company_id = "ACME-001"
            return True
        mock_lookup.side_effect = side_effect

        mock_activity.return_value.data = {"activities": [{"type": "Call"}]}
        mock_activity.return_value.sources = []

        mock_history.return_value.data = {"history": [{"type": "Note"}]}
        mock_history.return_value.sources = []

        mock_pipeline.return_value.data = {
            "opportunities": [{"name": "Deal 1"}],
            "summary": {"total": 100000},
        }
        mock_pipeline.return_value.sources = []

        ctx = IntentContext(
            question="what's happening with acme",
            resolved_company_id="ACME-001",
            days=90,
        )
        result = handle_company_status(ctx)

        assert result.company_data["found"] is True
        assert result.activities_data is not None
        assert result.history_data is not None
        assert result.pipeline_data is not None
        assert result.resolved_company_id == "ACME-001"

    @patch('backend.agent.fetch.handlers.company.lookup_company')
    def test_handles_company_not_found(self, mock_lookup):
        """Handles company not found."""
        # lookup_company returns False and sets company_data with found=False
        def side_effect(result, company_id):
            result.company_data = {"found": False, "query": company_id}
            return False
        mock_lookup.side_effect = side_effect

        ctx = IntentContext(
            question="what about xyz corp",
            resolved_company_id="XYZ-001",
            days=90,
        )
        result = handle_company_status(ctx)

        assert result.company_data["found"] is False
        assert result.activities_data is None


# =============================================================================
# handle_fallback Tests
# =============================================================================

class TestHandleFallback:
    """Tests for handle_fallback handler."""

    @patch('backend.agent.fetch.handlers.pipeline.tool_upcoming_renewals')
    def test_fetches_renewals_as_fallback(self, mock_tool, basic_context):
        """Fetches renewals as fallback."""
        mock_tool.return_value.data = {
            "renewals": [{"name": "Company 1"}],
        }
        mock_tool.return_value.sources = []

        result = handle_fallback(basic_context)

        assert result.renewals_data is not None


# =============================================================================
# dispatch_intent Tests
# =============================================================================

class TestDispatchIntent:
    """Tests for dispatch_intent function."""

    @patch('backend.agent.fetch.handlers.pipeline.tool_pipeline_summary')
    def test_dispatches_to_pipeline_summary(self, mock_tool, basic_context):
        """Dispatches pipeline_summary intent."""
        mock_tool.return_value.data = {"total_count": 0, "total_value": 0, "by_stage": [], "top_opportunities": []}
        mock_tool.return_value.sources = []

        result = dispatch_intent("pipeline_summary", basic_context)

        assert result is not None
        mock_tool.assert_called_once()

    @patch('backend.agent.fetch.handlers.pipeline.tool_upcoming_renewals')
    def test_dispatches_to_renewals(self, mock_tool, basic_context):
        """Dispatches renewals intent."""
        mock_tool.return_value.data = {"renewals": []}
        mock_tool.return_value.sources = []

        result = dispatch_intent("renewals", basic_context)

        assert result is not None
        mock_tool.assert_called_once()

    @patch('backend.agent.fetch.handlers.activity.tool_search_activities')
    def test_dispatches_activities_without_company(self, mock_tool):
        """Dispatches activities without company to activities handler."""
        from backend.agent.fetch.handlers.schemas import ToolResult
        mock_tool.return_value = ToolResult(data={"activities": []}, sources=[])

        ctx = IntentContext(
            question="show all calls",
            resolved_company_id=None,
            days=90,
        )
        result = dispatch_intent("activities", ctx)

        # Should call activities tool, not company lookup
        mock_tool.assert_called_once()
        assert result is not None

    @patch('backend.agent.fetch.handlers.handle_company_status')
    def test_dispatches_to_company_status_when_company_resolved(self, mock_handler):
        """Dispatches to company_status when company is resolved."""
        mock_handler.return_value = IntentResult()

        ctx = IntentContext(
            question="what about acme",
            resolved_company_id="ACME-001",
            days=90,
        )
        dispatch_intent("activities", ctx)

        mock_handler.assert_called_once()

    @patch('backend.agent.fetch.handlers.handle_fallback')
    def test_dispatches_to_fallback_for_unknown(self, mock_handler, basic_context):
        """Dispatches to fallback for unknown intent."""
        mock_handler.return_value = IntentResult()

        dispatch_intent("unknown_intent", basic_context)

        mock_handler.assert_called_once()


# =============================================================================
# INTENT_HANDLERS Tests
# =============================================================================

class TestIntentHandlers:
    """Tests for INTENT_HANDLERS mapping."""

    def test_has_expected_intents(self):
        """Has all expected intent handlers."""
        expected = [
            "pipeline_summary",
            "renewals",
            "deals_at_risk",
            "forecast",
            "contact_lookup",
            "contact_search",
            "company_search",
            "attachments",
            "activities",
            "analytics",
            "company_status",
            "pipeline",
            "history",
            "account_context",
            "general",
        ]
        for intent in expected:
            assert intent in INTENT_HANDLERS

    def test_handlers_are_callable(self):
        """All handlers are callable."""
        for handler in INTENT_HANDLERS.values():
            assert callable(handler)


# =============================================================================
# _safe_extend Tests
# =============================================================================

class TestSafeExtend:
    """Tests for _safe_extend helper function."""

    def test_extends_list_with_valid_source(self):
        """Extends target list with source list."""
        target = [1, 2]
        source = [3, 4]
        safe_extend(target, source)
        assert target == [1, 2, 3, 4]

    def test_handles_none_source(self):
        """Handles None source gracefully."""
        target = [1, 2]
        safe_extend(target, None)
        assert target == [1, 2]

    def test_handles_empty_source(self):
        """Handles empty source list."""
        target = [1, 2]
        safe_extend(target, [])
        assert target == [1, 2]

    def test_modifies_target_in_place(self):
        """Modifies target list in place."""
        target = ["a"]
        source = ["b", "c"]
        original_id = id(target)
        safe_extend(target, source)
        assert id(target) == original_id
        assert target == ["a", "b", "c"]

    def test_works_with_source_objects(self):
        """Works with Source objects."""
        target = []
        source = [Source(id="src1", type="company", label="Acme")]
        safe_extend(target, source)
        assert len(target) == 1
        assert target[0].id == "src1"


# =============================================================================
# handle_analytics Tests (coverage improvement)
# =============================================================================


class TestHandleAnalytics:
    """Tests for handle_analytics handler."""

    @patch('backend.agent.fetch.handlers.activity.tool_analytics')
    def test_handles_analytics_request(self, mock_tool, basic_context):
        """Handles analytics request."""
        mock_tool.return_value.data = {
            "metric": "activity_count",
            "total": 50,
            "breakdown": [{"type": "Call", "count": 20}],
        }
        mock_tool.return_value.sources = []

        from backend.agent.fetch.handlers.activity import handle_analytics
        result = handle_analytics(basic_context)

        assert result.analytics_data is not None
        assert result.raw_data["analytics"] is not None

    @patch('backend.agent.fetch.handlers.activity.tool_analytics')
    def test_passes_detected_metric_to_tool(self, mock_tool):
        """Passes detected metric to analytics tool."""
        mock_tool.return_value.data = {}
        mock_tool.return_value.sources = []

        from backend.agent.fetch.handlers.activity import handle_analytics
        # Must include "activit" for activity_type to be returned
        ctx = IntentContext(
            question="how many call activities were made last month",
            resolved_company_id=None,
            days=30,
        )
        handle_analytics(ctx)

        call_kwargs = mock_tool.call_args[1]
        assert call_kwargs["metric"] == "activity_count"
        assert call_kwargs["activity_type"] == "call"


class TestDetectAnalyticsMetric:
    """Tests for _detect_analytics_metric helper."""

    def test_detects_count_query(self):
        """Detects activity count queries."""
        from backend.agent.fetch.handlers.activity import _detect_analytics_metric

        metric, group_by, activity_type = _detect_analytics_metric("how many activities")
        assert metric == "activity_count"

    def test_detects_breakdown_query(self):
        """Detects activity breakdown queries."""
        from backend.agent.fetch.handlers.activity import _detect_analytics_metric

        metric, group_by, activity_type = _detect_analytics_metric(
            "show activity breakdown by type"
        )
        assert metric == "activity_breakdown"

    def test_detects_contact_role_breakdown(self):
        """Detects contact role breakdown queries."""
        from backend.agent.fetch.handlers.activity import _detect_analytics_metric

        metric, group_by, activity_type = _detect_analytics_metric(
            "contact breakdown by role"
        )
        assert metric == "contact_breakdown"
        assert group_by == "role"

    def test_detects_specific_activity_type(self):
        """Detects specific activity type in query."""
        from backend.agent.fetch.handlers.activity import _detect_analytics_metric

        # Must include "activit" for activity_type to be returned
        metric, group_by, activity_type = _detect_analytics_metric(
            "how many call activities were made"
        )
        assert activity_type == "call"

    def test_detects_email_activity(self):
        """Detects email activity type."""
        from backend.agent.fetch.handlers.activity import _detect_analytics_metric

        metric, group_by, activity_type = _detect_analytics_metric(
            "count of email activities"
        )
        assert activity_type == "email"

    def test_detects_meeting_activity(self):
        """Detects meeting activity type."""
        from backend.agent.fetch.handlers.activity import _detect_analytics_metric

        # Must include "activit" for activity_type to be returned
        metric, group_by, activity_type = _detect_analytics_metric(
            "total number of meeting activities scheduled"
        )
        assert activity_type == "meeting"

    def test_default_to_breakdown(self):
        """Defaults to activity breakdown for unclear queries."""
        from backend.agent.fetch.handlers.activity import _detect_analytics_metric

        metric, group_by, activity_type = _detect_analytics_metric(
            "show me the activity stats"
        )
        assert metric == "activity_breakdown"
        assert group_by == "type"

    def test_detects_comparison_keywords(self):
        """Detects comparison keywords for breakdown."""
        from backend.agent.fetch.handlers.activity import _detect_analytics_metric

        metric, group_by, activity_type = _detect_analytics_metric(
            "which activity type is most common"
        )
        assert metric == "activity_breakdown"

    def test_detects_percentage_keyword(self):
        """Detects percentage keyword for breakdown."""
        from backend.agent.fetch.handlers.activity import _detect_analytics_metric

        metric, group_by, activity_type = _detect_analytics_metric(
            "what percentage of activities are calls"
        )
        assert "breakdown" in metric


# =============================================================================
# handle_forecast_accuracy Tests (coverage improvement)
# =============================================================================


class TestHandleForecastAccuracy:
    """Tests for handle_forecast_accuracy handler."""

    @patch('backend.agent.fetch.handlers.pipeline.tool_forecast_accuracy')
    def test_fetches_forecast_accuracy(self, mock_tool, basic_context):
        """Fetches forecast accuracy data."""
        mock_tool.return_value.data = {
            "overall_win_rate": 45.5,
            "total_won": 10,
            "total_lost": 12,
            "by_owner": {},
        }
        mock_tool.return_value.sources = [Source(id="accuracy", type="pipeline", label="Accuracy")]

        from backend.agent.fetch.handlers.pipeline import handle_forecast_accuracy
        result = handle_forecast_accuracy(basic_context)

        assert result.pipeline_data["overall_win_rate"] == 45.5
        assert result.raw_data["pipeline_summary"] is not None
        assert result.raw_data["analytics"] is not None
        assert len(result.sources) == 1

    @patch('backend.agent.fetch.handlers.pipeline.tool_forecast_accuracy')
    def test_passes_owner_filter(self, mock_tool):
        """Passes owner filter to tool."""
        mock_tool.return_value.data = {"overall_win_rate": 50}
        mock_tool.return_value.sources = []

        from backend.agent.fetch.handlers.pipeline import handle_forecast_accuracy
        ctx = IntentContext(
            question="what is my win rate",
            resolved_company_id=None,
            days=90,
            owner="John Doe",
        )
        handle_forecast_accuracy(ctx)

        call_kwargs = mock_tool.call_args[1]
        assert call_kwargs["owner"] == "John Doe"


# =============================================================================
# handle_pipeline_summary with owner Tests
# =============================================================================


class TestHandlePipelineSummaryWithOwner:
    """Tests for handle_pipeline_summary with owner filter."""

    @patch('backend.agent.fetch.handlers.pipeline.tool_pipeline_by_owner')
    def test_uses_owner_filtered_tool(self, mock_tool):
        """Uses owner-filtered pipeline tool when owner is set."""
        mock_tool.return_value.data = {
            "summary": {"total_count": 5, "total_value": 100000},
            "opportunities": [{"name": "Deal 1"}, {"name": "Deal 2"}],
        }
        mock_tool.return_value.sources = []

        ctx = IntentContext(
            question="show my pipeline",
            resolved_company_id=None,
            days=90,
            owner="Jane Smith",
        )
        result = handle_pipeline_summary(ctx)

        mock_tool.assert_called_once_with(owner="Jane Smith")
        assert result.pipeline_data["summary"]["total_count"] == 5


# =============================================================================
# handle_deals_at_risk Extended Tests
# =============================================================================


class TestHandleDealsAtRiskExtended:
    """Extended tests for handle_deals_at_risk handler."""

    @patch('backend.agent.fetch.handlers.company.tool_accounts_needing_attention')
    @patch('backend.agent.fetch.handlers.pipeline.tool_upcoming_renewals')
    @patch('backend.agent.fetch.handlers.pipeline.tool_deals_at_risk')
    def test_fetches_all_risk_data(self, mock_risk, mock_renewals, mock_accounts, basic_context):
        """Fetches deals at risk, renewals, and accounts needing attention."""
        mock_risk.return_value.data = {
            "deals": [{"name": "Stalled Deal", "days_in_stage": 60}],
            "count": 1,
            "total_value": 50000,
        }
        mock_risk.return_value.sources = []

        mock_renewals.return_value.data = {
            "renewals": [{"name": "Expiring Soon"}],
        }
        mock_renewals.return_value.sources = []

        mock_accounts.return_value.data = {
            "accounts": [{"name": "Needs Attention Corp"}],
            "count": 1,
        }
        mock_accounts.return_value.sources = []

        result = handle_deals_at_risk(basic_context)

        assert result.pipeline_data is not None
        assert result.renewals_data is not None
        assert len(result.raw_data["opportunities"]) == 1
        assert len(result.raw_data["companies"]) == 1
        assert result.raw_data["pipeline_summary"]["at_risk_count"] == 1
        assert result.raw_data["pipeline_summary"]["accounts_needing_attention"] == 1

    @patch('backend.agent.fetch.handlers.company.tool_accounts_needing_attention')
    @patch('backend.agent.fetch.handlers.pipeline.tool_upcoming_renewals')
    @patch('backend.agent.fetch.handlers.pipeline.tool_deals_at_risk')
    def test_passes_owner_filter(self, mock_risk, mock_renewals, mock_accounts):
        """Passes owner filter to all tools."""
        mock_risk.return_value.data = {"deals": [], "count": 0, "total_value": 0}
        mock_risk.return_value.sources = []
        mock_renewals.return_value.data = {"renewals": []}
        mock_renewals.return_value.sources = []
        mock_accounts.return_value.data = {"accounts": [], "count": 0}
        mock_accounts.return_value.sources = []

        ctx = IntentContext(
            question="show my at-risk deals",
            resolved_company_id=None,
            days=90,
            owner="Bob Wilson",
        )
        handle_deals_at_risk(ctx)

        # Check owner was passed to deals at risk tool
        mock_risk.assert_called_once_with(owner="Bob Wilson")
        # Check owner was passed to renewals tool
        mock_renewals.assert_called_once()
        assert mock_renewals.call_args[1]["owner"] == "Bob Wilson"


# =============================================================================
# handle_forecast Extended Tests
# =============================================================================


class TestHandleForecastExtended:
    """Extended tests for handle_forecast handler."""

    @patch('backend.agent.fetch.handlers.pipeline.tool_forecast')
    def test_passes_owner_filter(self, mock_tool):
        """Passes owner filter to forecast tool."""
        mock_tool.return_value.data = {
            "total_pipeline": 100000,
            "total_weighted": 45000,
            "top_opportunities": [],
        }
        mock_tool.return_value.sources = []

        ctx = IntentContext(
            question="show my forecast",
            resolved_company_id=None,
            days=90,
            owner="Alice Manager",
        )
        result = handle_forecast(ctx)

        mock_tool.assert_called_once_with(owner="Alice Manager")
        assert result.pipeline_data["total_weighted"] == 45000

    @patch('backend.agent.fetch.handlers.pipeline.tool_forecast')
    def test_includes_top_opportunities(self, mock_tool, basic_context):
        """Includes top opportunities in raw data."""
        mock_tool.return_value.data = {
            "total_pipeline": 200000,
            "total_weighted": 90000,
            "top_opportunities": [
                {"name": "Big Deal", "value": 100000},
                {"name": "Medium Deal", "value": 50000},
            ],
        }
        mock_tool.return_value.sources = []

        result = handle_forecast(basic_context)

        assert len(result.raw_data["opportunities"]) == 2


# =============================================================================
# handle_renewals with Owner Tests
# =============================================================================


class TestHandleRenewalsWithOwner:
    """Tests for handle_renewals with owner filter."""

    @patch('backend.agent.fetch.handlers.pipeline.tool_upcoming_renewals')
    def test_passes_owner_filter(self, mock_tool):
        """Passes owner filter to renewals tool."""
        mock_tool.return_value.data = {"renewals": []}
        mock_tool.return_value.sources = []

        from backend.agent.fetch.handlers.pipeline import handle_renewals
        ctx = IntentContext(
            question="my renewals this quarter",
            resolved_company_id=None,
            days=90,
            owner="Manager Mike",
        )
        handle_renewals(ctx)

        call_kwargs = mock_tool.call_args[1]
        assert call_kwargs["owner"] == "Manager Mike"
