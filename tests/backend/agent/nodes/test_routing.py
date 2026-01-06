"""
Tests for backend/agent/nodes/routing.py.

Tests the routing node for question analysis and parameter extraction.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

os.environ["MOCK_LLM"] = "1"


# =============================================================================
# Route Node Tests
# =============================================================================


class TestRouteNode:
    """Tests for route_node function."""

    @patch('backend.agent.route.node.route_question')
    def test_route_node_returns_router_result(self, mock_route):
        """Returns router result in state."""
        from backend.agent.route.node import route_node

        mock_result = MagicMock()
        mock_result.mode_used = "data"
        mock_result.company_id = "ACME-001"
        mock_result.intent = "company_status"
        mock_result.days = 30
        mock_route.return_value = mock_result

        state = {
            "question": "What's happening with Acme?",
            "messages": [],
        }

        result = route_node(state)

        assert result["router_result"] is mock_result
        assert result["mode_used"] == "data"
        assert result["resolved_company_id"] == "ACME-001"
        assert result["intent"] == "company_status"

    @patch('backend.agent.route.node.route_question')
    def test_route_node_passes_conversation_history(self, mock_route):
        """Passes formatted conversation history to router."""
        from backend.agent.route.node import route_node

        mock_result = MagicMock()
        mock_result.mode_used = "data"
        mock_result.company_id = None
        mock_result.intent = "general"
        mock_result.days = 30
        mock_route.return_value = mock_result

        state = {
            "question": "What about their pipeline?",
            "messages": [
                {"role": "user", "content": "Tell me about Acme"},
                {"role": "assistant", "content": "Acme is a company..."},
            ],
        }

        route_node(state)

        call_kwargs = mock_route.call_args[1]
        assert "conversation_history" in call_kwargs
        assert len(call_kwargs["conversation_history"]) > 0

    @patch('backend.agent.route.node.route_question')
    def test_route_node_records_latency(self, mock_route):
        """Records routing latency."""
        from backend.agent.route.node import route_node

        mock_result = MagicMock()
        mock_result.mode_used = "data"
        mock_result.company_id = None
        mock_result.intent = "general"
        mock_result.days = 30
        mock_route.return_value = mock_result

        state = {
            "question": "Test question",
            "messages": [],
        }

        result = route_node(state)

        assert "router_latency_ms" in result
        assert result["router_latency_ms"] >= 0

    @patch('backend.agent.route.node.route_question')
    def test_route_node_returns_steps(self, mock_route):
        """Returns steps for progress tracking."""
        from backend.agent.route.node import route_node

        mock_result = MagicMock()
        mock_result.mode_used = "data"
        mock_result.company_id = None
        mock_result.intent = "general"
        mock_result.days = 30
        mock_route.return_value = mock_result

        state = {
            "question": "Test",
            "messages": [],
        }

        result = route_node(state)

        assert "steps" in result
        assert len(result["steps"]) > 0
        assert result["steps"][0]["id"] == "router"
        assert result["steps"][0]["status"] == "done"

    @patch('backend.agent.route.node.route_question')
    def test_route_node_handles_exception_with_data_fallback(self, mock_route):
        """Handles exception and uses data fallback for data keywords."""
        from backend.agent.route.node import route_node

        mock_route.side_effect = Exception("Router error")

        state = {
            "question": "What's happening with our company accounts?",
            "messages": [],
        }

        result = route_node(state)

        # Should fallback to data mode for data-related keywords
        assert result["mode_used"] == "data"
        assert result["intent"] == "general"
        assert "error" in result
        assert result["steps"][0]["status"] == "error"

    @patch('backend.agent.route.node.route_question')
    def test_route_node_handles_exception_with_docs_fallback(self, mock_route):
        """Handles exception and uses docs fallback for non-data keywords."""
        from backend.agent.route.node import route_node

        mock_route.side_effect = Exception("Router error")

        state = {
            "question": "How do I use this feature?",
            "messages": [],
        }

        result = route_node(state)

        # Should fallback to docs mode for non-data keywords
        assert result["mode_used"] == "docs"
        assert "error" in result

    @patch('backend.agent.route.node.route_question')
    def test_route_node_handles_missing_mode_used(self, mock_route):
        """Handles router result missing mode_used."""
        from backend.agent.route.node import route_node

        mock_result = MagicMock()
        mock_result.mode_used = None  # Missing mode
        mock_route.return_value = mock_result

        state = {
            "question": "What about customer data?",
            "messages": [],
        }

        result = route_node(state)

        # Should fallback due to invalid result
        assert "error" in result
        assert result["steps"][0]["status"] == "error"

    @patch('backend.agent.route.node.route_question')
    def test_route_node_uses_default_days_when_none(self, mock_route):
        """Uses default days when router returns None."""
        from backend.agent.route.node import route_node

        mock_result = MagicMock()
        mock_result.mode_used = "data"
        mock_result.company_id = None
        mock_result.intent = "general"
        mock_result.days = None  # No days specified
        mock_route.return_value = mock_result

        state = {
            "question": "Show me activities",
            "messages": [],
        }

        result = route_node(state)

        # Should use default days from config
        assert result["days"] > 0

    @patch('backend.agent.route.node.route_question')
    def test_route_node_uses_default_intent_when_none(self, mock_route):
        """Uses 'general' intent when router returns None."""
        from backend.agent.route.node import route_node

        mock_result = MagicMock()
        mock_result.mode_used = "data"
        mock_result.company_id = None
        mock_result.intent = None  # No intent
        mock_result.days = 30
        mock_route.return_value = mock_result

        state = {
            "question": "Test",
            "messages": [],
        }

        result = route_node(state)

        assert result["intent"] == "general"

    @patch('backend.agent.route.node.route_question')
    def test_route_node_handles_empty_messages(self, mock_route):
        """Handles state with no messages."""
        from backend.agent.route.node import route_node

        mock_result = MagicMock()
        mock_result.mode_used = "data"
        mock_result.company_id = None
        mock_result.intent = "general"
        mock_result.days = 30
        mock_route.return_value = mock_result

        state = {
            "question": "Test question",
            # No messages key
        }

        result = route_node(state)

        # Should work without messages
        assert result["mode_used"] == "data"
        call_kwargs = mock_route.call_args[1]
        assert call_kwargs["conversation_history"] == ""

    @patch('backend.agent.route.node.route_question')
    def test_route_node_fallback_detects_pipeline_keyword(self, mock_route):
        """Fallback detects 'pipeline' keyword for data mode."""
        from backend.agent.route.node import route_node

        mock_route.side_effect = Exception("Error")

        state = {
            "question": "Show me the pipeline status",
            "messages": [],
        }

        result = route_node(state)

        assert result["mode_used"] == "data"

    @patch('backend.agent.route.node.route_question')
    def test_route_node_fallback_detects_renewal_keyword(self, mock_route):
        """Fallback detects 'renewal' keyword for data mode."""
        from backend.agent.route.node import route_node

        mock_route.side_effect = Exception("Error")

        state = {
            "question": "Any renewal coming up?",
            "messages": [],
        }

        result = route_node(state)

        assert result["mode_used"] == "data"

