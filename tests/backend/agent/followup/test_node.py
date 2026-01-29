"""Tests for backend.agent.followup.node module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestFollowupNode:
    """Tests for followup_node function."""

    @patch("backend.agent.followup.node.get_connection")
    @patch("backend.agent.followup.node.generate_follow_up_suggestions")
    @patch("backend.agent.followup.node.format_conversation_for_prompt")
    def test_passes_answer_from_state(
        self,
        mock_format: MagicMock,
        mock_generate: MagicMock,
        mock_conn: MagicMock,
    ):
        """followup_node passes state answer to the generator."""
        from backend.agent.followup.node import followup_node

        mock_format.return_value = ""
        mock_generate.return_value = ["Q1?", "Q2?", "Q3?"]
        mock_connection = mock_conn.return_value

        state = {
            "question": "What deals does Acme have?",
            "answer": "Acme has 3 deals.",
            "messages": [],
        }

        result = followup_node(state)

        mock_generate.assert_called_once_with(
            question="What deals does Acme have?",
            answer="Acme has 3 deals.",
            conversation_history="",
            sql_results=None,
            conn=mock_connection,
        )
        assert result == {"follow_up_suggestions": ["Q1?", "Q2?", "Q3?"]}

    @patch("backend.agent.followup.node.get_connection")
    @patch("backend.agent.followup.node.generate_follow_up_suggestions")
    @patch("backend.agent.followup.node.format_conversation_for_prompt")
    def test_missing_answer_defaults_empty(
        self,
        mock_format: MagicMock,
        mock_generate: MagicMock,
        mock_conn: MagicMock,
    ):
        """followup_node passes empty string when answer not in state."""
        from backend.agent.followup.node import followup_node

        mock_format.return_value = ""
        mock_generate.return_value = ["Q1?", "Q2?", "Q3?"]
        mock_connection = mock_conn.return_value

        state = {"question": "Test?", "messages": []}

        followup_node(state)

        mock_generate.assert_called_once_with(
            question="Test?",
            answer="",
            conversation_history="",
            sql_results=None,
            conn=mock_connection,
        )

    @patch("backend.agent.followup.node.get_connection")
    @patch("backend.agent.followup.node.generate_follow_up_suggestions")
    @patch("backend.agent.followup.node.format_conversation_for_prompt")
    def test_passes_sql_results_from_state(
        self,
        mock_format: MagicMock,
        mock_generate: MagicMock,
        mock_conn: MagicMock,
    ):
        """followup_node passes sql_results from state to generator."""
        from backend.agent.followup.node import followup_node

        mock_format.return_value = ""
        mock_generate.return_value = ["Q1?", "Q2?", "Q3?"]
        mock_connection = mock_conn.return_value

        sql_results = {"data": [{"company_id": "C001", "name": "Acme"}]}
        state = {
            "question": "Test?",
            "answer": "Answer.",
            "messages": [],
            "sql_results": sql_results,
        }

        followup_node(state)

        mock_generate.assert_called_once_with(
            question="Test?",
            answer="Answer.",
            conversation_history="",
            sql_results=sql_results,
            conn=mock_connection,
        )

    @patch("backend.agent.followup.node.get_connection")
    @patch("backend.agent.followup.node.generate_follow_up_suggestions")
    @patch("backend.agent.followup.node.format_conversation_for_prompt")
    def test_filters_empty_suggestions(
        self,
        mock_format: MagicMock,
        mock_generate: MagicMock,
        mock_conn: MagicMock,
    ):
        """followup_node filters out empty/whitespace suggestions."""
        from backend.agent.followup.node import followup_node

        mock_format.return_value = ""
        mock_generate.return_value = ["Q1?", "", "  ", "Q2?"]

        state = {"question": "Test?", "answer": "Answer.", "messages": []}

        result = followup_node(state)

        assert result == {"follow_up_suggestions": ["Q1?", "Q2?"]}

    @patch("backend.agent.followup.node.get_connection")
    @patch("backend.agent.followup.node.generate_follow_up_suggestions")
    @patch("backend.agent.followup.node.format_conversation_for_prompt")
    def test_exception_returns_empty_list(
        self, mock_format: MagicMock, mock_generate: MagicMock, mock_conn: MagicMock
    ):
        """followup_node catches exceptions and returns empty list."""
        from backend.agent.followup.node import followup_node

        mock_format.return_value = ""
        mock_generate.side_effect = ValueError("LLM error")

        state = {"question": "Test?", "answer": "Answer.", "messages": []}

        result = followup_node(state)

        assert result == {"follow_up_suggestions": []}
