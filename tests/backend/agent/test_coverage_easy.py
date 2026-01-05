"""
Coverage tests for easy-to-test uncovered lines.

Covers edge cases in:
- handlers/common.py (non-list data)
- llm_helpers.py (mock suggestions, format_available_data)
- question_tree.py (validation)
- tools/activity.py (analytics edge cases)
- tools/company.py (search filters)
- eval/models.py (properties)
- conversation.py (message extraction)
- datastore/analytics.py (group not found)
- datastore/core.py (empty results)
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json


# =============================================================================
# handlers/common.py - line 95: non-list data handling
# =============================================================================


class TestHandlersCommon:
    """Tests for handlers/common.py edge cases."""

    def test_apply_tool_result_non_list_data(self):
        """Test apply_tool_result when data value is not a list."""
        from backend.agent.handlers.common import apply_tool_result, IntentResult, empty_raw_data
        from backend.agent.core.schemas import ToolResult, Source

        result = IntentResult(raw_data=empty_raw_data())
        tool_result = ToolResult(
            data={"summary": {"total": 100, "average": 25}},  # Non-list value
            sources=[Source(type="test", id="test1", label="Test")],
        )

        apply_tool_result(
            result=result,
            tool_result=tool_result,
            data_attr="analytics_data",
            raw_data_key="summary",
        )

        # Should store non-list data directly without slicing
        assert result.raw_data["summary"] == {"total": 100, "average": 25}
        assert result.analytics_data == {"summary": {"total": 100, "average": 25}}


# =============================================================================
# llm_helpers.py - lines 345, 373, 375: mock suggestions and format_available_data
# =============================================================================


class TestLlmHelpersMockSuggestions:
    """Tests for _get_mock_suggestions and _format_available_data."""

    def test_mock_suggestions_with_renewals(self):
        """Test mock suggestions include renewal question when renewals exist."""
        from backend.agent.llm.helpers import _get_mock_suggestions

        suggestions = _get_mock_suggestions(
            company_name="Acme Corp",
            available_data={"renewals": 1}
        )

        assert any("renewal" in s.lower() for s in suggestions)
        assert len(suggestions) <= 3

    def test_format_available_data_with_pipeline_and_docs(self):
        """Test _format_available_data includes pipeline and docs."""
        from backend.agent.llm.helpers import _format_available_data

        result = _format_available_data(
            data={
                "contacts": 5,
                "pipeline_summary": True,
                "docs": 10,
            },
            company_name="Acme"
        )

        assert "Pipeline" in result
        assert "Documentation" in result
        assert "10" in result  # docs count

    def test_format_available_data_empty_after_filtering(self):
        """Test _format_available_data with all zero counts."""
        from backend.agent.llm.helpers import _format_available_data

        result = _format_available_data(
            data={"contacts": 0, "activities": 0},
            company_name=None
        )

        assert "No specific data available" in result


# =============================================================================
# question_tree.py - get_paths_for_role behavior
# =============================================================================


class TestQuestionTreeBehavior:
    """Tests for question_tree public API behavior."""

    def test_get_paths_for_role_terminal_node(self):
        """Test get_paths_for_role handles terminal nodes correctly."""
        from backend.agent.question_tree import get_paths_for_role

        # This naturally exercises terminal node handling
        paths = get_paths_for_role()

        # All paths should end (not infinite)
        assert len(paths) > 0
        # Each path should have at least one question
        for path in paths:
            assert len(path) >= 1


# =============================================================================
# tools/activity.py - lines 84, 160, 173-174: analytics edge cases
# =============================================================================


class TestToolsActivityEdgeCases:
    """Tests for tools/activity.py edge cases."""

    def test_search_activities_with_company_id(self):
        """Test search_activities includes company in search description."""
        from backend.agent.tools.activity import tool_search_activities

        # Use real datastore - this exercises the company_id filter path
        result = tool_search_activities(company_id="ACME-MFG")

        # Company filter should be in the data filters
        assert result.data["filters"]["company_id"] == "ACME-MFG"

    def test_analytics_activity_count_with_activity_type(self):
        """Test activity_count metric with activity_type filter."""
        from backend.agent.tools.activity import tool_analytics

        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = None
        mock_ds.get_activity_count_by_filter.return_value = {"count": 5}

        result = tool_analytics(
            metric="activity_count",
            activity_type="call",
            days=30,
            datastore=mock_ds
        )

        mock_ds.get_activity_count_by_filter.assert_called_once()
        call_args = mock_ds.get_activity_count_by_filter.call_args
        assert call_args.kwargs.get("activity_type") == "call"

    def test_analytics_unknown_metric(self):
        """Test analytics returns error for unknown metric."""
        from backend.agent.tools.activity import tool_analytics

        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = None

        result = tool_analytics(
            metric="invalid_metric",
            datastore=mock_ds
        )

        assert "error" in result.data
        assert "Unknown metric" in result.data["error"]
        assert "available_metrics" in result.data


# =============================================================================
# tools/company.py - lines 62, 139: search filter descriptions
# =============================================================================


class TestToolsCompanyEdgeCases:
    """Tests for tools/company.py edge cases."""

    def test_search_companies_with_industry_filter(self):
        """Test search_companies includes industry in filters."""
        from backend.agent.tools.company import tool_search_companies

        # Use real datastore - this exercises the industry filter path
        result = tool_search_companies(industry="Technology")

        # Industry filter should be in the data
        assert result.data["filters"]["industry"] == "Technology"

    def test_search_contacts_with_job_title_filter(self):
        """Test search_contacts includes job_title in filters."""
        from backend.agent.tools.company import tool_search_contacts

        # Use real datastore - this exercises the job_title filter path
        result = tool_search_contacts(job_title="Engineer")

        # Job title filter should be in the data
        assert result.data["filters"]["job_title"] == "Engineer"


# =============================================================================
# eval/models.py - lines 209, 248, 253: dataclass properties
# =============================================================================


class TestEvalModelsProperties:
    """Tests for eval/models.py property methods."""

    def test_flow_step_result_passed_property(self):
        """Test FlowStepResult.passed property."""
        from backend.agent.eval.models import FlowStepResult

        # Passing case: has_answer=True, both scores=1
        passing = FlowStepResult(
            question="test?",
            answer="answer",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=1,
            grounded_score=1,
        )
        assert passing.passed is True

        # Failing case: has_answer but scores are 0
        failing = FlowStepResult(
            question="test?",
            answer="answer",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0,
            grounded_score=1,
        )
        assert failing.passed is False

        # Failing case: no answer
        no_answer = FlowStepResult(
            question="test?",
            answer="",
            latency_ms=100,
            has_answer=False,
            has_sources=False,
            relevance_score=1,
            grounded_score=1,
        )
        assert no_answer.passed is False

    def test_flow_eval_results_path_pass_rate_zero_division(self):
        """Test path_pass_rate handles zero paths tested."""
        from backend.agent.eval.models import FlowEvalResults

        # Zero paths tested - should return 0.0, not raise
        results = FlowEvalResults(
            total_paths=0,
            paths_tested=0,
            paths_passed=0,
            paths_failed=0,
            total_questions=0,
            questions_passed=0,
            questions_failed=0,
        )
        assert results.path_pass_rate == 0.0

    def test_flow_eval_results_question_pass_rate_zero_division(self):
        """Test question_pass_rate handles zero questions."""
        from backend.agent.eval.models import FlowEvalResults

        # Zero questions - should return 0.0, not raise
        results = FlowEvalResults(
            total_paths=0,
            paths_tested=0,
            paths_passed=0,
            paths_failed=0,
            total_questions=0,
            questions_passed=0,
            questions_failed=0,
        )
        assert results.question_pass_rate == 0.0


# =============================================================================
# conversation.py - lines 55-58: message extraction from checkpoint
# =============================================================================


class TestConversationMessageExtraction:
    """Tests for conversation.py message extraction."""

    def test_get_session_messages_with_messages(self):
        """Test get_session_messages returns messages from checkpoint state."""
        from backend.agent.session.conversation import get_session_messages

        with patch("backend.agent.session.conversation.get_session_state") as mock_get_state:
            mock_get_state.return_value = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }

            messages = get_session_messages("test-session-123")

            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"

    def test_get_session_messages_empty_messages_list(self):
        """Test get_session_messages with empty messages in state."""
        from backend.agent.session.conversation import get_session_messages

        with patch("backend.agent.session.conversation.get_session_state") as mock_get_state:
            mock_get_state.return_value = {"messages": []}

            messages = get_session_messages("test-session-456")

            assert messages == []


# =============================================================================
# datastore/analytics.py - line 283: group not found error
# =============================================================================


class TestAnalyticsGroupNotFound:
    """Tests for analytics.py group not found error."""

    def test_get_pipeline_by_group_not_found(self):
        """Test get_pipeline_by_group returns error for non-existent group."""
        from backend.agent.datastore import get_datastore

        ds = get_datastore()
        result = ds.get_pipeline_by_group(group_id="NONEXISTENT-GROUP-12345")

        assert "error" in result
        assert "not found" in result["error"]


# =============================================================================
# datastore/core.py - lines 135-136: empty query result
# =============================================================================


class TestDatastoreCoreEmptyResult:
    """Tests for datastore/core.py empty result handling."""

    def test_fetch_all_dicts_empty_result(self):
        """Test _fetch_all_dicts returns empty list for no results."""
        from backend.agent.datastore import get_datastore

        ds = get_datastore()
        
        # Query for a company that doesn't exist
        result = ds.search_companies(query="ZZZNONEXISTENT99999")

        # Should return empty list, not error
        assert result == []


# =============================================================================
# tools/activity.py - line 160: activity_type in label (need real datastore)
# =============================================================================


class TestActivityTypeLabel:
    """Test activity_type appears in analytics label."""

    def test_analytics_activity_count_label_includes_type(self):
        """Test that activity_count with activity_type includes it in source label."""
        from backend.agent.tools.activity import tool_analytics

        # Use real datastore to hit the actual code path
        result = tool_analytics(
            metric="activity_count",
            activity_type="email",
            days=30,
        )

        # The source label should include the activity type
        assert len(result.sources) > 0
        assert "email" in result.sources[0].label.lower() or "type" in result.sources[0].label.lower()

    def test_analytics_activity_count_with_company_name(self):
        """Test that activity_count with company includes company name in label (line 160)."""
        from backend.agent.tools.activity import tool_analytics

        # Use real company that exists in the datastore
        result = tool_analytics(
            metric="activity_count",
            company_id="ACME-MFG",  # Real company ID
            days=30,
        )

        # The source label should include "for <company_name>"
        assert len(result.sources) > 0
        label = result.sources[0].label.lower()
        assert "for" in label or "acme" in label


# =============================================================================
# tools/company.py - line 62: industry in label (need real datastore)
# =============================================================================


class TestCompanyIndustryLabel:
    """Test industry appears in search label."""

    def test_search_companies_label_includes_industry(self):
        """Test that search_companies with industry includes it in source label."""
        from backend.agent.tools.company import tool_search_companies

        # Use real datastore to hit the actual code path
        result = tool_search_companies(industry="Manufacturing")

        # The source label should include the industry filter
        assert len(result.sources) > 0
        label = result.sources[0].label.lower()
        assert "industry" in label or "manufacturing" in label