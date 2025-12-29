"""
Tests for backend/agent/intent_handlers.py - intent dispatch and handlers.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from backend.agent.intent_handlers import (
    IntentContext,
    IntentResult,
    dispatch_intent,
    handle_pipeline_summary,
    handle_renewals,
    handle_contacts,
    handle_company_search,
    handle_attachments,
    handle_activities,
    handle_company_status,
    handle_fallback,
    handle_deals_at_risk,
    handle_forecast,
    INTENT_HANDLERS,
    _empty_raw_data,
    _safe_extend,
)
from backend.agent.schemas import Source


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
        data = _empty_raw_data()
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

    @patch('backend.agent.intent_handlers.tool_pipeline_summary')
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

    @patch('backend.agent.intent_handlers.tool_upcoming_renewals')
    def test_fetches_renewals(self, mock_tool, basic_context):
        """Fetches renewal data."""
        mock_tool.return_value.data = {
            "renewals": [{"name": "Company 1", "renewal_date": "2025-03-01"}],
        }
        mock_tool.return_value.sources = []

        result = handle_renewals(basic_context)

        assert result.renewals_data is not None
        assert len(result.raw_data["renewals"]) == 1

    @patch('backend.agent.intent_handlers.tool_company_lookup')
    @patch('backend.agent.intent_handlers.tool_upcoming_renewals')
    def test_includes_company_when_resolved(self, mock_renewals, mock_company):
        """Includes company data when company_id is resolved."""
        mock_company.return_value.data = {
            "found": True,
            "company": {"company_id": "ACME-001", "name": "Acme Corp"},
        }
        mock_company.return_value.sources = []
        mock_renewals.return_value.data = {"renewals": []}
        mock_renewals.return_value.sources = []

        ctx = IntentContext(
            question="renewals for acme",
            resolved_company_id="ACME-001",
            days=90,
        )
        result = handle_renewals(ctx)

        assert result.company_data is not None
        assert result.company_data["found"] is True


# =============================================================================
# handle_contacts Tests
# =============================================================================

class TestHandleContacts:
    """Tests for handle_contacts handler."""

    @patch('backend.agent.intent_handlers.tool_search_contacts')
    def test_fetches_contacts(self, mock_tool, basic_context):
        """Fetches contact data."""
        mock_tool.return_value.data = {
            "contacts": [{"name": "John Doe", "role": "CEO"}],
        }
        mock_tool.return_value.sources = []

        result = handle_contacts(basic_context)

        assert result.contacts_data is not None
        assert len(result.raw_data["contacts"]) == 1

    @patch('backend.agent.intent_handlers.tool_search_contacts')
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

    @patch('backend.agent.intent_handlers.tool_search_companies')
    def test_searches_companies(self, mock_tool, basic_context):
        """Searches companies."""
        mock_tool.return_value.data = {
            "companies": [{"name": "Tech Corp", "segment": "Enterprise"}],
        }
        mock_tool.return_value.sources = []

        result = handle_company_search(basic_context)

        assert result.company_data is not None
        assert len(result.raw_data["companies"]) == 1

    @patch('backend.agent.intent_handlers.tool_search_companies')
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

    @patch('backend.agent.intent_handlers.tool_search_attachments')
    def test_searches_attachments(self, mock_tool, basic_context):
        """Searches attachments."""
        mock_tool.return_value.data = {
            "attachments": [{"name": "Proposal.pdf"}],
        }
        mock_tool.return_value.sources = []

        result = handle_attachments(basic_context)

        assert result.attachments_data is not None
        assert len(result.raw_data["attachments"]) == 1

    @patch('backend.agent.intent_handlers.tool_search_attachments')
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

    @patch('backend.agent.intent_handlers.tool_search_activities')
    def test_searches_activities(self, mock_tool, basic_context):
        """Searches activities."""
        mock_tool.return_value.data = {
            "activities": [{"type": "Call", "subject": "Check-in"}],
        }
        mock_tool.return_value.sources = []

        result = handle_activities(basic_context)

        assert result.activities_data is not None
        assert len(result.raw_data["activities"]) == 1

    @patch('backend.agent.intent_handlers.tool_search_activities')
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

    @patch('backend.agent.intent_handlers.tool_pipeline')
    @patch('backend.agent.intent_handlers.tool_recent_history')
    @patch('backend.agent.intent_handlers.tool_recent_activity')
    @patch('backend.agent.intent_handlers.tool_company_lookup')
    def test_fetches_full_company_data(
        self, mock_lookup, mock_activity, mock_history, mock_pipeline
    ):
        """Fetches all company data when found."""
        mock_lookup.return_value.data = {
            "found": True,
            "company": {"company_id": "ACME-001", "name": "Acme Corp"},
        }
        mock_lookup.return_value.sources = []

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

    @patch('backend.agent.intent_handlers.tool_company_lookup')
    def test_handles_company_not_found(self, mock_lookup):
        """Handles company not found."""
        mock_lookup.return_value.data = {"found": False}
        mock_lookup.return_value.sources = []

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

    @patch('backend.agent.intent_handlers.tool_upcoming_renewals')
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

    @patch('backend.agent.intent_handlers.tool_pipeline_summary')
    def test_dispatches_to_pipeline_summary(self, mock_tool, basic_context):
        """Dispatches pipeline_summary intent."""
        mock_tool.return_value.data = {"total_count": 0, "total_value": 0, "by_stage": [], "top_opportunities": []}
        mock_tool.return_value.sources = []

        result = dispatch_intent("pipeline_summary", basic_context)

        assert result is not None
        mock_tool.assert_called_once()

    @patch('backend.agent.intent_handlers.tool_upcoming_renewals')
    def test_dispatches_to_renewals(self, mock_tool, basic_context):
        """Dispatches renewals intent."""
        mock_tool.return_value.data = {"renewals": []}
        mock_tool.return_value.sources = []

        result = dispatch_intent("renewals", basic_context)

        assert result is not None
        mock_tool.assert_called_once()

    @patch('backend.agent.intent_handlers.handle_activities')
    def test_dispatches_activities_without_company(self, mock_handler):
        """Dispatches activities without company to activities handler."""
        mock_handler.return_value = IntentResult()

        ctx = IntentContext(
            question="show all calls",
            resolved_company_id=None,
            days=90,
        )
        dispatch_intent("activities", ctx)

        mock_handler.assert_called_once()

    @patch('backend.agent.intent_handlers.handle_company_status')
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

    @patch('backend.agent.intent_handlers.handle_fallback')
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
        _safe_extend(target, source)
        assert target == [1, 2, 3, 4]

    def test_handles_none_source(self):
        """Handles None source gracefully."""
        target = [1, 2]
        _safe_extend(target, None)
        assert target == [1, 2]

    def test_handles_empty_source(self):
        """Handles empty source list."""
        target = [1, 2]
        _safe_extend(target, [])
        assert target == [1, 2]

    def test_modifies_target_in_place(self):
        """Modifies target list in place."""
        target = ["a"]
        source = ["b", "c"]
        original_id = id(target)
        _safe_extend(target, source)
        assert id(target) == original_id
        assert target == ["a", "b", "c"]

    def test_works_with_source_objects(self):
        """Works with Source objects."""
        target = []
        source = [Source(id="src1", type="company", label="Acme")]
        _safe_extend(target, source)
        assert len(target) == 1
        assert target[0].id == "src1"
