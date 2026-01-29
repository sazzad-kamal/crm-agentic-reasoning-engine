"""Tests for backend.agent.followup.suggester module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.agent.followup.suggester import (
    FollowUpSuggestions,
    generate_follow_up_suggestions,
)


class TestGenerateFollowUpSuggestions:
    """Tests for generate_follow_up_suggestions function."""

    @patch("backend.agent.followup.tree.get_follow_ups")
    def test_hardcoded_tree_returns_suggestions(self, mock_tree: MagicMock):
        """Hardcoded tree match returns suggestions without calling LLM."""
        mock_tree.return_value = ["Follow-up 1?", "Follow-up 2?", "Follow-up 3?"]

        result = generate_follow_up_suggestions(
            question="What deals does Acme have?",
            answer="Acme has 3 deals.",
        )

        assert result == ["Follow-up 1?", "Follow-up 2?", "Follow-up 3?"]
        mock_tree.assert_called_once_with("What deals does Acme have?")

    @patch("backend.agent.followup.suggester._get_followup_chain")
    def test_llm_fallback_when_no_tree_match(self, mock_get_chain: MagicMock):
        """LLM fallback is used when hardcoded tree has no match."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = FollowUpSuggestions(
            suggestions=["LLM Q1?", "LLM Q2?", "LLM Q3?"]
        )

        result = generate_follow_up_suggestions(
            question="Some novel question?",
            answer="The answer is 42.",
            use_hardcoded_tree=False,
        )

        assert result == ["LLM Q1?", "LLM Q2?", "LLM Q3?"]
        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert call_kwargs["question"] == "Some novel question?"
        assert "The answer is 42." in call_kwargs["answer_section"]

    @patch("backend.agent.followup.suggester._get_followup_chain")
    def test_llm_fallback_empty_answer(self, mock_get_chain: MagicMock):
        """LLM fallback with empty answer passes empty answer_section."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = FollowUpSuggestions(
            suggestions=["Q1?", "Q2?", "Q3?"]
        )

        generate_follow_up_suggestions(
            question="Test?",
            answer="",
            use_hardcoded_tree=False,
        )

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert call_kwargs["answer_section"] == ""

    @patch("backend.agent.followup.suggester._get_followup_chain")
    def test_llm_fallback_with_conversation_history(self, mock_get_chain: MagicMock):
        """LLM fallback includes conversation history when provided."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = FollowUpSuggestions(
            suggestions=["Q1?", "Q2?", "Q3?"]
        )

        generate_follow_up_suggestions(
            question="Test?",
            answer="Answer.",
            conversation_history="User: Hello\nAssistant: Hi",
            use_hardcoded_tree=False,
        )

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert "RECENT CONVERSATION" in call_kwargs["conversation_history_section"]

    @patch("backend.agent.followup.suggester._get_followup_chain")
    def test_llm_exception_returns_empty(self, mock_get_chain: MagicMock):
        """LLM failure returns empty list."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.side_effect = ValueError("API error")

        result = generate_follow_up_suggestions(
            question="Test?",
            use_hardcoded_tree=False,
        )

        assert result == []

    @patch("backend.agent.followup.tree.get_follow_ups")
    @patch("backend.agent.followup.suggester._get_followup_chain")
    def test_tree_miss_falls_through_to_llm(
        self, mock_get_chain: MagicMock, mock_tree: MagicMock
    ):
        """When tree returns empty, falls through to LLM."""
        mock_tree.return_value = []
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = FollowUpSuggestions(
            suggestions=["A?", "B?", "C?"]
        )

        result = generate_follow_up_suggestions(
            question="Unknown question?",
            answer="Some answer.",
        )

        assert result == ["A?", "B?", "C?"]
        mock_tree.assert_called_once()
        mock_chain.invoke.assert_called_once()

    @patch("backend.agent.followup.tree.get_follow_ups")
    def test_use_hardcoded_tree_false_skips_tree(self, mock_tree: MagicMock):
        """Setting use_hardcoded_tree=False skips tree lookup entirely."""
        with patch("backend.agent.followup.suggester._get_followup_chain") as mock_get_chain:
            mock_chain = mock_get_chain.return_value
            mock_chain.invoke.return_value = FollowUpSuggestions(
                suggestions=["X?", "Y?", "Z?"]
            )

            generate_follow_up_suggestions(
                question="Test?",
                use_hardcoded_tree=False,
            )

        mock_tree.assert_not_called()

    @patch("backend.agent.followup.suggester._get_followup_chain")
    def test_schema_section_included_in_llm_call(self, mock_get_chain: MagicMock):
        """LLM fallback includes schema section in chain invoke."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = FollowUpSuggestions(
            suggestions=["Q1?", "Q2?", "Q3?"]
        )

        generate_follow_up_suggestions(
            question="Test?",
            answer="Answer.",
            use_hardcoded_tree=False,
        )

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert "DATABASE SCHEMA" in call_kwargs["schema_section"]
        assert "CREATE TABLE" in call_kwargs["schema_section"]

    @patch("backend.agent.followup.suggester.get_entity_context")
    @patch("backend.agent.followup.suggester._get_followup_chain")
    def test_entity_context_included_when_sql_results_provided(
        self, mock_get_chain: MagicMock, mock_entity_ctx: MagicMock
    ):
        """Entity context is included when sql_results and conn are provided."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = FollowUpSuggestions(
            suggestions=["Q1?", "Q2?", "Q3?"]
        )
        mock_entity_ctx.return_value = "- Acme Corp (company): 3 contacts"

        conn = MagicMock()
        sql_results = {"data": [{"company_id": "C001"}]}

        generate_follow_up_suggestions(
            question="Test?",
            answer="Answer.",
            use_hardcoded_tree=False,
            sql_results=sql_results,
            conn=conn,
        )

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert "ENTITY CONTEXT" in call_kwargs["entity_context_section"]
        assert "Acme Corp" in call_kwargs["entity_context_section"]
        mock_entity_ctx.assert_called_once_with(sql_results, conn)

    @patch("backend.agent.followup.suggester._get_followup_chain")
    def test_no_entity_context_without_sql_results(self, mock_get_chain: MagicMock):
        """Entity context is empty when no sql_results provided."""
        mock_chain = mock_get_chain.return_value
        mock_chain.invoke.return_value = FollowUpSuggestions(
            suggestions=["Q1?", "Q2?", "Q3?"]
        )

        generate_follow_up_suggestions(
            question="Test?",
            answer="Answer.",
            use_hardcoded_tree=False,
        )

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert call_kwargs["entity_context_section"] == ""
