"""
Tests for answer and follow-up generation nodes.

Tests answer_node (backend/agent/answer/node.py) and
followup_node (backend/agent/followup/node.py).
"""

import os

import pytest
from unittest.mock import patch, MagicMock

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

        mock_chain.return_value = ("This is the answer.", 150)

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

        mock_chain.return_value = ("Response text.", 100)

        state = {
            "question": "Tell me about Acme",
            "messages": [{"role": "user", "content": "Previous message"}],
            "sql_results": {},
        }

        result = answer_node(state)

        assert len(result["messages"]) == 3
        assert result["messages"][-2]["role"] == "user"
        assert result["messages"][-2]["content"] == "Tell me about Acme"
        assert result["messages"][-1]["role"] == "assistant"
        assert result["messages"][-1]["content"] == "Response text."

    @patch('backend.agent.answer.node.call_answer_chain')
    def test_answer_node_handles_empty_answer(self, mock_chain):
        """Uses fallback when LLM returns empty answer."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = ("", 100)

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

        mock_chain.return_value = ("Full answer", 100)

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
            "sql_results": {"company_info": [{"name": "Acme Corp"}]},
        }

        result = followup_node(state)

        assert "follow_up_suggestions" in result
        assert len(result["follow_up_suggestions"]) == 2

    def test_followup_node_respects_disabled_setting(self):
        """Returns empty when suggestions disabled."""
        from backend.agent.followup.node import followup_node

        with patch('backend.agent.followup.node._ENABLE_FOLLOW_UP_SUGGESTIONS', False):
            state = {
                "question": "Test",
                "messages": [],
            }

            result = followup_node(state)

            assert result["follow_up_suggestions"] == []

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    def test_followup_node_limits_suggestions(self, mock_generate):
        """Limits suggestions to _MAX_FOLLOWUP_SUGGESTIONS."""
        from backend.agent.followup.node import followup_node

        mock_generate.return_value = [
            "Suggestion 1",
            "Suggestion 2",
            "Suggestion 3",
            "Suggestion 4",
        ]

        with patch('backend.agent.followup.node._MAX_FOLLOWUP_SUGGESTIONS', 2):
            state = {
                "question": "Test",
                "messages": [],
                "sql_results": {},
            }

            result = followup_node(state)

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
            "sql_results": {},
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
            "sql_results": {},
        }

        result = followup_node(state)

        assert result["follow_up_suggestions"] == []

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    def test_followup_node_extracts_company_name_from_sql_results(self, mock_generate):
        """Extracts company name from sql_results."""
        from backend.agent.followup.node import followup_node

        mock_generate.return_value = ["What about their pipeline?"]

        state = {
            "question": "Tell me about Acme",
            "messages": [],
            "sql_results": {
                "company_info": [{"name": "Acme Manufacturing", "company_id": "ACME-001"}],
            },
        }

        followup_node(state)

        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["company_name"] == "Acme Manufacturing"

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    def test_followup_node_builds_available_data_from_sql_results(self, mock_generate):
        """Builds available_data counts from sql_results."""
        from backend.agent.followup.node import followup_node

        mock_generate.return_value = []

        state = {
            "question": "Test",
            "messages": [],
            "sql_results": {
                "contacts": [{"name": "John"}, {"name": "Jane"}],
                "activities": [{"type": "Call"}],
                "opportunities": [],
                "history": [{"type": "Note"}],
            },
        }

        followup_node(state)

        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["available_data"]["contacts"] == 2
        assert call_kwargs["available_data"]["activities"] == 1

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    def test_followup_node_counts_non_list_data_as_one(self, mock_generate):
        """Non-list truthy data is counted as 1 in available_data."""
        from backend.agent.followup.node import followup_node

        mock_generate.return_value = []

        state = {
            "question": "Test",
            "messages": [],
            "sql_results": {
                "contacts": [{"name": "John"}],  # List - should count as 1
                "summary": "Pipeline summary text",  # String - should count as 1
                "total_value": 50000,  # Number - should count as 1
                "empty_field": "",  # Empty string - should not be counted
                "null_field": None,  # None - should not be counted
            },
        }

        followup_node(state)

        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["available_data"]["contacts"] == 1
        assert call_kwargs["available_data"]["summary"] == 1
        assert call_kwargs["available_data"]["total_value"] == 1
        assert "empty_field" not in call_kwargs["available_data"]
        assert "null_field" not in call_kwargs["available_data"]
