"""
Tests for backend/agent/route/node.py.

Tests the routing node for query planning.
"""

import os

import pytest
from unittest.mock import patch

os.environ["MOCK_LLM"] = "1"


class TestRouteNode:
    """Tests for route_node function."""

    @patch('backend.agent.route.node.get_sql_plan')
    def test_route_node_returns_sql_plan(self, mock_sql_planner):
        """Returns SQL plan in state."""
        from backend.agent.route.node import route_node
        from backend.agent.route.sql_planner import SQLPlan

        mock_sql_plan = SQLPlan(
            sql="SELECT * FROM companies",
            needs_rag=True
        )
        mock_sql_planner.return_value = mock_sql_plan

        state = {
            "question": "What's happening with Acme?",
            "messages": [],
        }

        result = route_node(state)

        assert result["sql_plan"] is not None
        assert result["sql_plan"].sql == "SELECT * FROM companies"
        assert result["needs_rag"] is True

    @patch('backend.agent.route.node.get_sql_plan')
    def test_route_node_passes_conversation_history(self, mock_sql_planner):
        """Passes formatted conversation history to planner."""
        from backend.agent.route.node import route_node
        from backend.agent.route.sql_planner import SQLPlan

        mock_sql_plan = SQLPlan(
            sql="SELECT * FROM opportunities",
            needs_rag=False
        )
        mock_sql_planner.return_value = mock_sql_plan

        state = {
            "question": "What about their pipeline?",
            "messages": [
                {"role": "user", "content": "Tell me about Acme"},
                {"role": "assistant", "content": "Acme is a company..."},
            ],
        }

        route_node(state)

        call_kwargs = mock_sql_planner.call_args[1]
        assert "conversation_history" in call_kwargs
        assert len(call_kwargs["conversation_history"]) > 0

    @patch('backend.agent.route.node.get_sql_plan')
    def test_route_node_handles_exception_with_fallback(self, mock_sql_planner):
        """Handles exception and uses fallback SQL plan."""
        from backend.agent.route.node import route_node

        mock_sql_planner.side_effect = Exception("Planner error")

        state = {
            "question": "What's happening with our company accounts?",
            "messages": [],
        }

        result = route_node(state)

        assert "sql_plan" in result
        assert result["needs_rag"] is False
        assert "error" in result

    @patch('backend.agent.route.node.get_sql_plan')
    def test_route_node_handles_empty_messages(self, mock_sql_planner):
        """Handles state with no messages."""
        from backend.agent.route.node import route_node
        from backend.agent.route.sql_planner import SQLPlan

        mock_sql_plan = SQLPlan(
            sql="SELECT * FROM companies",
            needs_rag=False
        )
        mock_sql_planner.return_value = mock_sql_plan

        state = {
            "question": "Test question",
            # No messages key
        }

        result = route_node(state)

        call_kwargs = mock_sql_planner.call_args[1]
        assert call_kwargs["conversation_history"] == ""
