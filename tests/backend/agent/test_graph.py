"""
Tests for backend.agent.graph module.

Tests the LangGraph workflow construction and execution.
"""

import os
from unittest.mock import patch

import pytest

# Set mock mode
os.environ["MOCK_LLM"] = "1"

from langgraph.graph import END

from backend.agent.graph import (
    ACTION_NODE,
    ANSWER_NODE,
    COMPARE_NODE,
    EXPORT_NODE,
    FETCH_NODE,
    FOLLOWUP_NODE,
    HEALTH_NODE,
    PLANNER_NODE,
    SUPERVISOR_NODE,
    TREND_NODE,
    _route_after_answer,
    _route_after_supervisor,
    agent_graph,
    build_thread_config,
)


def _invoke_agent(question: str, session_id: str | None = None) -> dict:
    """Helper to invoke agent for tests."""
    state = {"question": question, "session_id": session_id, "sources": []}
    config = build_thread_config(session_id)
    return agent_graph.invoke(state, config=config)


class TestGraphConstruction:
    """Tests for graph building."""

    def test_agent_graph_is_compiled(self):
        """Test that agent_graph is a compiled graph."""
        assert agent_graph is not None
        assert hasattr(agent_graph, "invoke")

    def test_supervisor_node_exists(self):
        """Test that supervisor node is in the graph."""
        assert SUPERVISOR_NODE == "supervisor"


class TestRouteAfterSupervisor:
    """Tests for _route_after_supervisor conditional routing."""

    def test_routes_to_fetch_for_data_query(self):
        """Data queries should route to Fetch node."""
        state = {"intent": "data_query"}
        result = _route_after_supervisor(state)
        assert result == FETCH_NODE

    def test_routes_to_answer_for_clarify(self):
        """Clarify intent should route directly to Answer."""
        state = {"intent": "clarify"}
        result = _route_after_supervisor(state)
        assert result == ANSWER_NODE

    def test_routes_to_answer_for_help(self):
        """Help intent should route directly to Answer."""
        state = {"intent": "help"}
        result = _route_after_supervisor(state)
        assert result == ANSWER_NODE

    def test_defaults_to_fetch_when_no_intent(self):
        """Missing intent should default to Fetch (data_query behavior)."""
        state = {}
        result = _route_after_supervisor(state)
        assert result == FETCH_NODE

    def test_routes_to_compare_for_compare_intent(self):
        """Compare intent should route to Compare node."""
        state = {"intent": "compare"}
        result = _route_after_supervisor(state)
        assert result == COMPARE_NODE

    def test_routes_to_trend_for_trend_intent(self):
        """Trend intent should route to Trend node."""
        state = {"intent": "trend"}
        result = _route_after_supervisor(state)
        assert result == TREND_NODE

    def test_routes_to_planner_for_complex_intent(self):
        """Complex intent should route to Planner node."""
        state = {"intent": "complex"}
        result = _route_after_supervisor(state)
        assert result == PLANNER_NODE

    def test_routes_to_export_for_export_intent(self):
        """Export intent should route to Export node."""
        state = {"intent": "export"}
        result = _route_after_supervisor(state)
        assert result == EXPORT_NODE

    def test_routes_to_health_for_health_intent(self):
        """Health intent should route to Health node."""
        state = {"intent": "health"}
        result = _route_after_supervisor(state)
        assert result == HEALTH_NODE


class TestRouteAfterAnswer:
    """Tests for _route_after_answer conditional routing."""

    def test_routes_to_action_and_followup_when_data_present(self):
        """Fan out to action+followup when sql_results has data."""
        state = {"sql_results": {"data": [{"id": 1}]}, "intent": "data_query"}
        result = _route_after_answer(state)
        assert result == [ACTION_NODE, FOLLOWUP_NODE]

    def test_routes_to_end_when_no_data(self):
        """Skip action+followup when sql_results is empty."""
        state = {"sql_results": {}, "intent": "data_query"}
        result = _route_after_answer(state)
        assert result == END

    def test_routes_to_end_when_sql_results_missing(self):
        """Skip action+followup when sql_results key is missing."""
        state = {"intent": "data_query"}
        result = _route_after_answer(state)
        assert result == END

    def test_routes_to_end_when_data_is_empty_list(self):
        """Skip action+followup when data key exists but is empty."""
        state = {"sql_results": {"data": []}, "intent": "data_query"}
        result = _route_after_answer(state)
        assert result == END

    def test_routes_to_fetch_when_needs_more_data(self):
        """Loop back to Fetch when Answer signals needs_more_data."""
        state = {"needs_more_data": True, "intent": "data_query"}
        result = _route_after_answer(state)
        assert result == FETCH_NODE

    def test_routes_to_end_for_clarify_intent(self):
        """Clarify responses should end without action/followup."""
        state = {"intent": "clarify", "answer": "Could you clarify?"}
        result = _route_after_answer(state)
        assert result == END

    def test_routes_to_end_for_help_intent(self):
        """Help responses should end without action/followup."""
        state = {"intent": "help", "answer": "I can help you with..."}
        result = _route_after_answer(state)
        assert result == END

    def test_needs_more_data_takes_priority(self):
        """needs_more_data should trigger loop even with data present."""
        state = {
            "needs_more_data": True,
            "sql_results": {"data": [{"id": 1}]},
            "intent": "data_query",
        }
        result = _route_after_answer(state)
        assert result == FETCH_NODE

    def test_routes_to_action_followup_for_comparison_data(self):
        """Should route to action/followup when comparison results present."""
        state = {
            "sql_results": {"comparison": {"entity_a": "Q1", "entity_b": "Q2"}},
            "intent": "compare",
        }
        result = _route_after_answer(state)
        assert result == [ACTION_NODE, FOLLOWUP_NODE]

    def test_routes_to_action_followup_for_trend_data(self):
        """Should route to action/followup when trend results present."""
        state = {
            "sql_results": {"trend_analysis": {"direction": "increasing"}},
            "intent": "trend",
        }
        result = _route_after_answer(state)
        assert result == [ACTION_NODE, FOLLOWUP_NODE]

    def test_routes_to_action_followup_for_export_data(self):
        """Should route to action/followup when export results present."""
        state = {
            "sql_results": {"export": {"status": "success"}},
            "intent": "export",
        }
        result = _route_after_answer(state)
        assert result == [ACTION_NODE, FOLLOWUP_NODE]

    def test_routes_to_action_followup_for_health_data(self):
        """Should route to action/followup when health results present."""
        state = {
            "sql_results": {"health_analysis": {"score": 85}},
            "intent": "health",
        }
        result = _route_after_answer(state)
        assert result == [ACTION_NODE, FOLLOWUP_NODE]

    def test_routes_to_action_followup_for_aggregated_data(self):
        """Should route to action/followup when planner aggregated results present."""
        state = {
            "sql_results": {"aggregated": {"data": {}}},
            "intent": "complex",
        }
        result = _route_after_answer(state)
        assert result == [ACTION_NODE, FOLLOWUP_NODE]

    def test_needs_more_data_only_loops_for_data_query(self):
        """needs_more_data should only loop for data_query intent."""
        state = {
            "needs_more_data": True,
            "intent": "compare",  # Not data_query
        }
        result = _route_after_answer(state)
        # Should not loop back, instead goes to END
        assert result == END


# =============================================================================
# Integration Tests (with mocked LLM)
# =============================================================================

class TestGraphIntegration:
    """Integration tests for the graph with mocked LLM."""

    @pytest.mark.integration
    @patch("backend.agent.action.suggester.call_action_chain")
    @patch("backend.agent.answer.answerer.call_answer_chain")
    @patch("backend.agent.sql.planner.get_sql_plan")
    def test_graph_execution(self, mock_planner, mock_answer_chain, mock_action_chain):
        """Test graph execution with company query."""
        from backend.agent.sql.planner import SQLPlan

        mock_planner.return_value = SQLPlan(
            sql="SELECT * FROM companies WHERE name ILIKE '%acme%'",
        )
        mock_answer_chain.return_value = "Acme Manufacturing is doing well."
        mock_action_chain.return_value = None

        result = _invoke_agent("What's the status of Acme?")

        assert "answer" in result
