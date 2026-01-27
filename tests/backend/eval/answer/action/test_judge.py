"""Tests for backend.eval.answer.action.judge module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.eval.answer.action.judge import (
    ActionJudgeResult,
    judge_suggested_action,
)


class TestActionJudgeResult:
    """Tests for ActionJudgeResult model."""

    def test_action_judge_result_creation(self):
        """Test ActionJudgeResult can be created."""
        result = ActionJudgeResult(
            relevance=0.8,
            actionability=0.9,
            appropriateness=0.85,
            explanation="Good action",
        )
        assert result.relevance == 0.8
        assert result.actionability == 0.9
        assert result.appropriateness == 0.85
        assert result.explanation == "Good action"


class TestJudgeSuggestedAction:
    """Tests for judge_suggested_action function."""

    @patch("backend.eval.answer.action.judge.create_openai_chain")
    def test_judge_suggested_action_passing(self, mock_chain_fn: MagicMock):
        """Test judge returns passing result."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ActionJudgeResult(
            relevance=0.95,
            actionability=0.9,
            appropriateness=0.9,
            explanation="Action is relevant and actionable",
        )
        mock_chain_fn.return_value = mock_chain

        passed, rel, act, app, explanation = judge_suggested_action(
            question="What deals are closing?",
            answer="3 deals worth $50k",
            action="Schedule follow-up calls",
        )

        assert passed is True
        assert rel == 0.95
        assert act == 0.9
        assert app == 0.9
        assert "relevant" in explanation.lower()

    @patch("backend.eval.answer.action.judge.create_openai_chain")
    def test_judge_suggested_action_failing(self, mock_chain_fn: MagicMock):
        """Test judge returns failing result."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ActionJudgeResult(
            relevance=0.4,
            actionability=0.3,
            appropriateness=0.5,
            explanation="Action is vague and not actionable",
        )
        mock_chain_fn.return_value = mock_chain

        passed, rel, act, app, explanation = judge_suggested_action(
            question="What deals are closing?",
            answer="3 deals worth $50k",
            action="Follow up",
        )

        assert passed is False
        assert rel == 0.4
        assert act == 0.3
        assert app == 0.5

    @patch("backend.eval.answer.action.judge.create_openai_chain")
    def test_judge_suggested_action_error_propagates(self, mock_chain_fn: MagicMock):
        """Test judge lets exceptions propagate to caller."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = ValueError("API error")
        mock_chain_fn.return_value = mock_chain

        with pytest.raises(ValueError, match="API error"):
            judge_suggested_action(
                question="What deals are closing?",
                answer="3 deals worth $50k",
                action="Send email",
            )
