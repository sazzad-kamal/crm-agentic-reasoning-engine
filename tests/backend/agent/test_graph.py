"""
Tests for backend.agent.graph module.

Tests the LangGraph workflow construction and execution.
"""

import os
import pytest
from unittest.mock import patch

# Set mock mode
os.environ["MOCK_LLM"] = "1"

from backend.agent.graph import agent_graph, build_thread_config


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


# =============================================================================
# Integration Tests (with mocked LLM)
# =============================================================================

class TestGraphIntegration:
    """Integration tests for the graph with mocked LLM."""

    @pytest.mark.integration
    @patch("backend.agent.answer.llm.call_answer_chain")
    @patch("backend.agent.route.sql_planner.get_sql_plan")
    def test_graph_execution(self, mock_planner, mock_answer_chain):
        """Test graph execution with company query."""
        from backend.agent.route.sql_planner import SQLPlan

        mock_planner.return_value = SQLPlan(
            sql="SELECT * FROM companies WHERE name ILIKE '%acme%'",
            needs_rag=True,
        )
        mock_answer_chain.return_value = ("Acme Manufacturing is doing well.", 100)

        result = _invoke_agent("What's the status of Acme?")

        assert "answer" in result
