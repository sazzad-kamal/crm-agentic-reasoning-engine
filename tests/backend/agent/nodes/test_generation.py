"""
Tests for answer and follow-up generation nodes.

Tests answer_node (backend/agent/answer/node.py) and
followup_node (backend/agent/followup/node.py).
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

os.environ["MOCK_LLM"] = "1"


# =============================================================================
# Answer Node Tests
# =============================================================================


class TestAnswerNode:
    """Tests for answer_node function."""

    @patch('backend.agent.answer.node.call_answer_chain')
    def test_answer_node_synthesizes_response(self, mock_chain):
        """Synthesizes response from state data."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = "This is the answer."

        state = {
            "question": "What's happening with Acme?",
            "messages": [],
            "sql_results": {"company_info": [{"name": "Acme"}]},
        }

        result = answer_node(state)

        assert result["answer"] == "This is the answer."
        mock_chain.assert_called_once()

    @patch('backend.agent.answer.node.call_answer_chain')
    def test_answer_node_updates_messages(self, mock_chain):
        """Updates messages with user question and assistant answer."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = "Response text."

        state = {
            "question": "Tell me about Acme",
            "messages": [HumanMessage(content="Previous message")],
            "sql_results": {},
        }

        result = answer_node(state)

        # Node returns only new messages; add_messages reducer appends them
        assert len(result["messages"]) == 2
        assert result["messages"][0].type == "human"
        assert result["messages"][0].content == "Tell me about Acme"
        assert result["messages"][1].type == "ai"
        assert result["messages"][1].content == "Response text."

    @patch('backend.agent.answer.node.call_answer_chain')
    def test_answer_node_handles_empty_answer(self, mock_chain):
        """Uses fallback when LLM returns empty answer."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = ""

        state = {
            "question": "Test question",
            "messages": [],
            "sql_results": {},
        }

        result = answer_node(state)

        assert "apologize" in result["answer"].lower() or "rephrasing" in result["answer"].lower()

    @patch('backend.agent.answer.node.call_answer_chain')
    def test_answer_node_handles_exception(self, mock_chain):
        """Handles exceptions gracefully."""
        from backend.agent.answer.node import answer_node

        mock_chain.side_effect = Exception("LLM error")

        state = {
            "question": "Test question",
            "messages": [],
            "sql_results": {},
        }

        result = answer_node(state)

        assert "error" in result
        assert result["error"] == "LLM error"
        assert "error" in result["answer"].lower()

    @patch('backend.agent.answer.node.call_answer_chain')
    def test_answer_node_passes_sql_results(self, mock_chain):
        """Passes sql_results to answer chain."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = "Full answer"

        state = {
            "question": "Give me everything about Acme",
            "messages": [],
            "sql_results": {
                "company_info": [{"name": "Acme", "company_id": "ACME-001"}],
                "open_deals": [{"name": "Upgrade", "value": 50000}],
            },
            "rag_context": "Context notes",
        }

        answer_node(state)

        # Verify chain was called with sql_results
        call_kwargs = mock_chain.call_args[1]
        assert "question" in call_kwargs
        assert "sql_results" in call_kwargs
        assert call_kwargs["sql_results"]["company_info"][0]["name"] == "Acme"

# =============================================================================
# Action Node Tests
# =============================================================================


class TestActionNode:
    """Tests for action_node function."""

    @patch('backend.agent.action.node.call_action_chain')
    def test_action_node_suggests_action(self, mock_chain):
        """Suggests action when chain returns one."""
        from backend.agent.action.node import action_node

        mock_chain.return_value = "Schedule a call with Sarah Chen"

        state = {
            "question": "What deals does Acme have?",
            "answer": "Acme has 3 deals.",
        }

        result = action_node(state)

        assert result["suggested_action"] == "Schedule a call with Sarah Chen"
        mock_chain.assert_called_once()

    @patch('backend.agent.action.node.call_action_chain')
    def test_action_node_returns_none_when_no_action(self, mock_chain):
        """Returns None when chain returns None."""
        from backend.agent.action.node import action_node

        mock_chain.return_value = None

        state = {
            "question": "How many deals?",
            "answer": "There are 5 deals.",
        }

        result = action_node(state)

        assert result["suggested_action"] is None

    @patch('backend.agent.action.node.call_action_chain')
    def test_action_node_handles_exception(self, mock_chain):
        """Handles exceptions gracefully."""
        from backend.agent.action.node import action_node

        mock_chain.side_effect = Exception("LLM error")

        state = {
            "question": "Test?",
            "answer": "Answer.",
        }

        result = action_node(state)

        assert result["suggested_action"] is None

    @patch('backend.agent.action.node.call_action_chain')
    def test_action_node_skips_on_error_state(self, mock_chain):
        """Skips LLM call when state has error."""
        from backend.agent.action.node import action_node

        state = {
            "question": "Test?",
            "answer": "Error answer.",
            "error": "some error",
        }

        result = action_node(state)

        assert result["suggested_action"] is None
        mock_chain.assert_not_called()


# =============================================================================
# Follow-up Node Tests
# =============================================================================


class TestFollowupNode:
    """Tests for followup_node function."""

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    def test_followup_node_generates_suggestions(self, mock_generate):
        """Generates follow-up suggestions."""
        from backend.agent.followup.node import followup_node

        mock_generate.return_value = [
            "What about their contacts?",
            "Show me recent activities",
        ]

        state = {
            "question": "Tell me about Acme",
            "messages": [],
        }

        result = followup_node(state)

        assert "follow_up_suggestions" in result
        assert len(result["follow_up_suggestions"]) == 2

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    def test_followup_node_filters_empty_suggestions(self, mock_generate):
        """Filters out empty suggestions."""
        from backend.agent.followup.node import followup_node

        mock_generate.return_value = [
            "Valid suggestion",
            "",
            "  ",
            "Another valid one",
        ]

        state = {
            "question": "Test",
            "messages": [],
        }

        result = followup_node(state)

        # Only non-empty suggestions
        assert len(result["follow_up_suggestions"]) == 2

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    def test_followup_node_handles_exception(self, mock_generate):
        """Handles exceptions gracefully."""
        from backend.agent.followup.node import followup_node

        mock_generate.side_effect = Exception("Generation failed")

        state = {
            "question": "Test",
            "messages": [],
        }

        result = followup_node(state)

        assert result["follow_up_suggestions"] == []

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    def test_followup_node_passes_question_and_history(self, mock_generate):
        """Passes question and conversation history to suggester."""
        from backend.agent.followup.node import followup_node

        mock_generate.return_value = ["Follow-up question?"]

        state = {
            "question": "Tell me about Acme",
            "messages": [
                HumanMessage(content="Previous question"),
                AIMessage(content="Previous answer"),
            ],
        }

        followup_node(state)

        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["question"] == "Tell me about Acme"
        assert "conversation_history" in call_kwargs
