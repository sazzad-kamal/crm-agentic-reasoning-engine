"""
Tests for answer and follow-up generation nodes.

Tests answer_node (backend/agent/nodes/answer.py) and
followup_node (backend/agent/nodes/followup.py).
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
            "company_data": {"found": True, "company": {"name": "Acme"}},
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
            "company_data": None,
            "resolved_company_id": "ACME-001",
        }

        result = answer_node(state)

        assert len(result["messages"]) == 3
        assert result["messages"][-2]["role"] == "user"
        assert result["messages"][-2]["content"] == "Tell me about Acme"
        assert result["messages"][-1]["role"] == "assistant"
        assert result["messages"][-1]["content"] == "Response text."

    @patch('backend.agent.answer.node.call_not_found_chain')
    def test_answer_node_handles_company_not_found(self, mock_chain):
        """Uses not-found chain when company not found."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = ("I couldn't find that company.", 80)

        state = {
            "question": "What about XYZ Corp?",
            "messages": [],
            "company_data": {
                "found": False,
                "query": "xyz corp",
                "close_matches": [{"name": "XY Inc", "company_id": "XY-001"}],
            },
        }

        result = answer_node(state)

        mock_chain.assert_called_once()
        assert "company" in result["answer"].lower() or len(result["answer"]) > 0

    @patch('backend.agent.answer.node.call_answer_chain')
    def test_answer_node_handles_empty_answer(self, mock_chain):
        """Uses fallback when LLM returns empty answer."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = ("", 100)

        state = {
            "question": "Test question",
            "messages": [],
            "company_data": None,
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
            "company_data": None,
        }

        result = answer_node(state)

        assert "error" in result
        assert result["error"] == "LLM error"
        assert "error" in result["answer"].lower()

    @patch('backend.agent.answer.node.call_answer_chain')
    def test_answer_node_records_latency(self, mock_chain):
        """Records answer latency."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = ("Answer text", 200)

        state = {
            "question": "Test",
            "messages": [],
            "company_data": None,
        }

        result = answer_node(state)

        assert "answer_latency_ms" in result
        assert result["llm_latency_ms"] == 200

    @patch('backend.agent.answer.node.call_answer_chain')
    def test_answer_node_returns_steps(self, mock_chain):
        """Returns steps for progress tracking."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = ("Answer", 100)

        state = {
            "question": "Test",
            "messages": [],
            "company_data": None,
        }

        result = answer_node(state)

        assert "steps" in result
        assert len(result["steps"]) > 0
        assert result["steps"][0]["id"] == "answer"
        assert result["steps"][0]["status"] == "done"

    @patch('backend.agent.answer.node.call_answer_chain')
    def test_answer_node_formats_all_sections(self, mock_chain):
        """Formats all data sections for context."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = ("Full answer", 100)

        state = {
            "question": "Give me everything about Acme",
            "messages": [],
            "company_data": {"found": True, "company": {"name": "Acme"}},
            "contacts_data": {"contacts": [{"name": "John"}]},
            "activities_data": {"activities": [{"type": "Call"}]},
            "history_data": {"history": []},
            "pipeline_data": {"opportunities": []},
            "renewals_data": {"renewals": []},
            "groups_data": {"groups": []},
            "attachments_data": {"attachments": []},
            "account_context_answer": "Context notes",
        }

        answer_node(state)

        # Verify chain was called with formatted sections
        call_kwargs = mock_chain.call_args[1]
        assert "question" in call_kwargs
        assert "company_section" in call_kwargs


# =============================================================================
# Follow-up Node Tests
# =============================================================================


class TestFollowupNode:
    """Tests for followup_node function."""

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    @patch('backend.agent.followup.node.get_config')
    def test_followup_node_generates_suggestions(self, mock_config, mock_generate):
        """Generates follow-up suggestions."""
        from backend.agent.followup.node import followup_node

        mock_config.return_value.enable_follow_up_suggestions = True
        mock_config.return_value.max_followup_suggestions = 3
        mock_generate.return_value = [
            "What about their contacts?",
            "Show me recent activities",
        ]

        state = {
            "question": "Tell me about Acme",
            "messages": [],
            "intent": "data",
            "company_data": {"found": True, "company": {"name": "Acme Corp"}},
            "raw_data": {"contacts": [], "activities": []},
        }

        result = followup_node(state)

        assert "follow_up_suggestions" in result
        assert len(result["follow_up_suggestions"]) == 2

    @patch('backend.agent.followup.node.get_config')
    def test_followup_node_respects_disabled_setting(self, mock_config):
        """Returns empty when suggestions disabled."""
        from backend.agent.followup.node import followup_node

        mock_config.return_value.enable_follow_up_suggestions = False

        state = {
            "question": "Test",
            "messages": [],
        }

        result = followup_node(state)

        assert result["follow_up_suggestions"] == []

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    @patch('backend.agent.followup.node.get_config')
    def test_followup_node_limits_suggestions(self, mock_config, mock_generate):
        """Limits suggestions to max_followup_suggestions."""
        from backend.agent.followup.node import followup_node

        mock_config.return_value.enable_follow_up_suggestions = True
        mock_config.return_value.max_followup_suggestions = 2
        mock_generate.return_value = [
            "Suggestion 1",
            "Suggestion 2",
            "Suggestion 3",
            "Suggestion 4",
        ]

        state = {
            "question": "Test",
            "messages": [],
            "intent": "auto",
            "company_data": None,
            "raw_data": {},
        }

        result = followup_node(state)

        assert len(result["follow_up_suggestions"]) == 2

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    @patch('backend.agent.followup.node.get_config')
    def test_followup_node_filters_empty_suggestions(self, mock_config, mock_generate):
        """Filters out empty suggestions."""
        from backend.agent.followup.node import followup_node

        mock_config.return_value.enable_follow_up_suggestions = True
        mock_config.return_value.max_followup_suggestions = 5
        mock_generate.return_value = [
            "Valid suggestion",
            "",
            "  ",
            "Another valid one",
        ]

        state = {
            "question": "Test",
            "messages": [],
            "intent": "data",
            "company_data": None,
            "raw_data": {},
        }

        result = followup_node(state)

        # Only non-empty suggestions
        assert len(result["follow_up_suggestions"]) == 2

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    @patch('backend.agent.followup.node.get_config')
    def test_followup_node_handles_exception(self, mock_config, mock_generate):
        """Handles exceptions gracefully."""
        from backend.agent.followup.node import followup_node

        mock_config.return_value.enable_follow_up_suggestions = True
        mock_generate.side_effect = Exception("Generation failed")

        state = {
            "question": "Test",
            "messages": [],
            "intent": "auto",
            "company_data": None,
            "raw_data": {},
        }

        result = followup_node(state)

        assert result["follow_up_suggestions"] == []
        assert result["steps"][0]["status"] == "error"

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    @patch('backend.agent.followup.node.get_config')
    def test_followup_node_records_latency(self, mock_config, mock_generate):
        """Records follow-up generation latency."""
        from backend.agent.followup.node import followup_node

        mock_config.return_value.enable_follow_up_suggestions = True
        mock_config.return_value.max_followup_suggestions = 3
        mock_generate.return_value = ["Suggestion"]

        state = {
            "question": "Test",
            "messages": [],
            "intent": "auto",
            "company_data": None,
            "raw_data": {},
        }

        result = followup_node(state)

        assert "followup_latency_ms" in result
        assert result["followup_latency_ms"] >= 0

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    @patch('backend.agent.followup.node.get_config')
    def test_followup_node_extracts_company_name(self, mock_config, mock_generate):
        """Extracts company name from company_data."""
        from backend.agent.followup.node import followup_node

        mock_config.return_value.enable_follow_up_suggestions = True
        mock_config.return_value.max_followup_suggestions = 3
        mock_generate.return_value = ["What about their pipeline?"]

        state = {
            "question": "Tell me about Acme",
            "messages": [],
            "intent": "data",
            "company_data": {
                "found": True,
                "company": {"name": "Acme Manufacturing", "company_id": "ACME-001"},
            },
            "raw_data": {},
        }

        followup_node(state)

        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["company_name"] == "Acme Manufacturing"

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    @patch('backend.agent.followup.node.get_config')
    def test_followup_node_builds_available_data(self, mock_config, mock_generate):
        """Builds available_data counts from raw_data."""
        from backend.agent.followup.node import followup_node

        mock_config.return_value.enable_follow_up_suggestions = True
        mock_config.return_value.max_followup_suggestions = 3
        mock_generate.return_value = []

        state = {
            "question": "Test",
            "messages": [],
            "intent": "data",
            "company_data": None,
            "raw_data": {
                "contacts": [{"name": "John"}, {"name": "Jane"}],
                "activities": [{"type": "Call"}],
                "opportunities": [],
                "history": [{"type": "Note"}],
                "renewals": [],
            },
        }

        followup_node(state)

        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["available_data"]["contacts"] == 2
        assert call_kwargs["available_data"]["activities"] == 1

    @patch('backend.agent.followup.node.generate_follow_up_suggestions')
    @patch('backend.agent.followup.node.get_config')
    def test_followup_node_returns_steps(self, mock_config, mock_generate):
        """Returns steps for progress tracking."""
        from backend.agent.followup.node import followup_node

        mock_config.return_value.enable_follow_up_suggestions = True
        mock_config.return_value.max_followup_suggestions = 3
        mock_generate.return_value = ["Suggestion"]

        state = {
            "question": "Test",
            "messages": [],
            "intent": "auto",
            "company_data": None,
            "raw_data": {},
        }

        result = followup_node(state)

        assert "steps" in result
        assert result["steps"][0]["id"] == "followup"
        assert result["steps"][0]["status"] == "done"
