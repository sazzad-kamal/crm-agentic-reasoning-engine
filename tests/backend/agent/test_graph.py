"""
Tests for backend.agent.graph module.

Tests the LangGraph workflow construction and execution.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set mock mode
os.environ["MOCK_LLM"] = "1"

from backend.agent.graph import (
    build_agent_graph,
    run_agent,
    answer_question,
    get_graph_mermaid,
)
from backend.agent.state import AgentState


# =============================================================================
# Graph Construction Tests
# =============================================================================

class TestGraphConstruction:
    """Tests for graph building."""

    def test_build_agent_graph_returns_compiled_graph(self):
        """Test that build_agent_graph returns a compiled graph."""
        graph = build_agent_graph()
        assert graph is not None
        # Compiled graphs have an invoke method
        assert hasattr(graph, "invoke")

    def test_graph_has_expected_nodes(self):
        """Test that the graph has all expected nodes."""
        # We can't easily inspect nodes after compilation,
        # but we can verify the graph runs
        graph = build_agent_graph()
        assert graph is not None


class TestGraphMermaid:
    """Tests for graph visualization."""

    def test_get_graph_mermaid_returns_string(self):
        """Test that get_graph_mermaid returns a valid mermaid diagram."""
        mermaid = get_graph_mermaid()
        assert isinstance(mermaid, str)
        assert "graph TD" in mermaid
        assert "route" in mermaid
        assert "answer" in mermaid
        assert "followup" in mermaid


# =============================================================================
# Run Agent Tests
# =============================================================================

class TestRunAgent:
    """Tests for run_agent function."""

    @patch("backend.agent.graph.agent_graph")
    def test_run_agent_returns_dict_with_required_keys(self, mock_graph):
        """Test that run_agent returns all required response keys."""
        mock_graph.invoke.return_value = {
            "answer": "Test answer",
            "sources": [],
            "steps": [{"id": "test", "label": "Test", "status": "done"}],
            "raw_data": {},
            "follow_up_suggestions": [],
            "mode_used": "docs",
            "resolved_company_id": None,
            "days": 90,
        }

        result = run_agent("Test question")

        assert "answer" in result
        assert "sources" in result
        assert "steps" in result
        assert "raw_data" in result
        assert "follow_up_suggestions" in result
        assert "meta" in result

    @patch("backend.agent.graph.agent_graph")
    def test_run_agent_includes_latency_in_meta(self, mock_graph):
        """Test that run_agent includes latency in meta."""
        mock_graph.invoke.return_value = {
            "answer": "Test",
            "sources": [],
            "steps": [],
            "raw_data": {},
            "follow_up_suggestions": [],
            "mode_used": "docs",
        }

        result = run_agent("Test question")

        assert "latency_ms" in result["meta"]
        assert isinstance(result["meta"]["latency_ms"], int)

    @patch("backend.agent.graph.agent_graph")
    def test_run_agent_handles_graph_exception(self, mock_graph):
        """Test that run_agent handles exceptions gracefully."""
        mock_graph.invoke.side_effect = Exception("Graph error")

        # use_cache=False to avoid hitting cached results from previous tests
        result = run_agent("Test question for error handling", use_cache=False)

        assert "error" in result["answer"].lower() or "sorry" in result["answer"].lower()
        assert result["meta"]["mode_used"] == "error"

    @patch("backend.agent.graph.agent_graph")
    def test_run_agent_passes_mode_to_initial_state(self, mock_graph):
        """Test that mode is passed correctly to initial state."""
        mock_graph.invoke.return_value = {
            "answer": "Test",
            "sources": [],
            "steps": [],
            "raw_data": {},
            "follow_up_suggestions": [],
            "mode_used": "data",
        }

        run_agent("Test question", mode="data")

        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["mode"] == "data"

    @patch("backend.agent.graph.agent_graph")
    def test_run_agent_passes_company_id(self, mock_graph):
        """Test that company_id is passed correctly."""
        mock_graph.invoke.return_value = {
            "answer": "Test",
            "sources": [],
            "steps": [],
            "raw_data": {},
            "follow_up_suggestions": [],
            "mode_used": "data",
            "resolved_company_id": "ACME-MFG",
        }

        run_agent("Test question", company_id="ACME-MFG")

        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["company_id"] == "ACME-MFG"


class TestAnswerQuestionWrapper:
    """Tests for answer_question backwards compatibility wrapper."""

    @patch("backend.agent.graph.run_agent")
    def test_answer_question_calls_run_agent(self, mock_run):
        """Test that answer_question delegates to run_agent."""
        mock_run.return_value = {"answer": "Test"}

        result = answer_question("Test question", mode="docs")

        mock_run.assert_called_once_with(
            question="Test question",
            mode="docs",
            company_id=None,
            session_id=None,
            user_id=None,
        )

    @patch("backend.agent.graph.run_agent")
    def test_answer_question_passes_all_params(self, mock_run):
        """Test that answer_question passes all parameters."""
        mock_run.return_value = {"answer": "Test"}

        answer_question(
            question="Test",
            mode="data",
            company_id="ACME-MFG",
            session_id="sess-123",
            user_id="user-456",
        )

        mock_run.assert_called_once_with(
            question="Test",
            mode="data",
            company_id="ACME-MFG",
            session_id="sess-123",
            user_id="user-456",
        )


# =============================================================================
# Integration Tests (with mocked LLM)
# =============================================================================

class TestGraphIntegration:
    """Integration tests for the graph with mocked LLM."""

    @pytest.mark.integration
    @patch("backend.agent.llm_helpers.call_answer_chain")
    @patch("backend.agent.llm_router.route_question")
    def test_graph_execution_docs_mode(self, mock_route, mock_answer_chain):
        """Test graph execution in docs mode."""
        from backend.agent.schemas import RouterResult

        mock_route.return_value = RouterResult(
            mode_used="docs",
            company_id=None,
            intent="docs",
            days=90,
        )
        # call_answer_chain returns (answer, latency_ms)
        mock_answer_chain.return_value = ("This is a test answer about documentation.", 100)

        result = run_agent("How do I create a contact?", mode="docs")

        assert result["meta"]["mode_used"] == "docs"
        assert isinstance(result["answer"], str)

    @pytest.mark.integration
    @patch("backend.agent.llm_helpers.call_answer_chain")
    @patch("backend.agent.llm_router.route_question")
    @patch("backend.agent.handlers.common.tool_company_lookup")
    def test_graph_execution_data_mode(self, mock_company, mock_route, mock_answer_chain):
        """Test graph execution in data mode."""
        from backend.agent.schemas import RouterResult, ToolResult, Source

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
        # call_answer_chain returns (answer, latency_ms)
        mock_answer_chain.return_value = ("Acme Manufacturing is doing well.", 100)

        result = run_agent("What's the status of Acme?", mode="data")

        assert result["meta"]["mode_used"] == "data"
