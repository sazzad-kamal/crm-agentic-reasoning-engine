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

from backend.agent.graph import ACTION_NODE, FOLLOWUP_NODE, _route_after_answer, agent_graph, build_thread_config


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


class TestRouteAfterAnswer:
    """Tests for _route_after_answer conditional routing."""

    def test_routes_to_action_and_followup_when_data_present(self):
        """Fan out to action+followup when sql_results has data."""
        state = {"sql_results": {"data": [{"id": 1}]}}
        result = _route_after_answer(state)
        assert result == [ACTION_NODE, FOLLOWUP_NODE]

    def test_routes_to_end_when_no_data(self):
        """Skip action+followup when sql_results is empty."""
        state = {"sql_results": {}}
        result = _route_after_answer(state)
        assert result == END

    def test_routes_to_end_when_sql_results_missing(self):
        """Skip action+followup when sql_results key is missing."""
        state = {}
        result = _route_after_answer(state)
        assert result == END

    def test_routes_to_end_when_data_is_empty_list(self):
        """Skip action+followup when data key exists but is empty."""
        state = {"sql_results": {"data": []}}
        result = _route_after_answer(state)
        assert result == END


# =============================================================================
# Integration Tests (with mocked LLM)
# =============================================================================

class TestGraphIntegration:
    """Integration tests for the graph with mocked LLM."""

    @pytest.mark.integration
    @patch("backend.agent.action.suggester.call_action_chain")
    @patch("backend.agent.answer.answerer.call_answer_chain")
    @patch("backend.agent.fetch.planner.get_sql_plan")
    def test_graph_execution(self, mock_planner, mock_answer_chain, mock_action_chain):
        """Test graph execution with company query."""
        from backend.agent.fetch.planner import SQLPlan

        mock_planner.return_value = SQLPlan(
            sql="SELECT * FROM companies WHERE name ILIKE '%acme%'",
        )
        mock_answer_chain.return_value = "Acme Manufacturing is doing well."
        mock_action_chain.return_value = None

        result = _invoke_agent("What's the status of Acme?")

        assert "answer" in result
