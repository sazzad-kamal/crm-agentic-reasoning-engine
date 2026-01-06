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
    @patch("backend.agent.route.router.route_question")
    def test_graph_execution_docs_mode(self, mock_route, mock_answer_chain):
        """Test graph execution in docs mode."""
        from backend.agent.core.schemas import RouterResult

        mock_route.return_value = RouterResult(
            mode_used="docs",
            company_id=None,
            intent="docs",
            days=90,
        )
        mock_answer_chain.return_value = ("This is a test answer about documentation.", 100)

        result = _invoke_agent("How do I create a contact?")

        assert "mode_used" in result
        assert isinstance(result.get("answer", ""), str)

    @pytest.mark.integration
    @patch("backend.agent.answer.llm.call_answer_chain")
    @patch("backend.agent.route.router.route_question")
    @patch("backend.agent.fetch.handlers.company.tool_company_lookup")
    def test_graph_execution_data_mode(self, mock_company, mock_route, mock_answer_chain):
        """Test graph execution in data mode."""
        from backend.agent.core.schemas import RouterResult, ToolResult, Source

        mock_route.return_value = RouterResult(
            mode_used="data",
            company_id="ACME-MFG",
            intent="company_status",
            days=90,
        )
        mock_company.return_value = ToolResult(
            data={"company_id": "ACME-MFG", "name": "Acme Manufacturing"},
            sources=[Source(type="company", id="ACME-MFG", label="Acme Manufacturing")],
        )
        mock_answer_chain.return_value = ("Acme Manufacturing is doing well.", 100)

        result = _invoke_agent("What's the status of Acme?")

        assert "mode_used" in result
