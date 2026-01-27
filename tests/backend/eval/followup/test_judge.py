"""Tests for backend.eval.followup.judge module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.eval.followup.judge import (
    FollowupJudgeResult,
    judge_followup_suggestions,
)


class TestFollowupJudgeResult:
    """Tests for FollowupJudgeResult model."""

    def test_followup_judge_result_creation(self):
        """Test FollowupJudgeResult can be created."""
        result = FollowupJudgeResult(
            question_relevance=0.8,
            answer_grounding=0.6,
            diversity=0.7,
            explanation="Good suggestions",
        )
        assert result.question_relevance == 0.8
        assert result.answer_grounding == 0.6
        assert result.diversity == 0.7
        assert result.explanation == "Good suggestions"


class TestJudgeFollowupSuggestions:
    """Tests for judge_followup_suggestions function."""

    @patch("backend.eval.followup.judge.create_openai_chain")
    def test_judge_followup_passing(self, mock_chain_fn: MagicMock):
        """Test judge returns passing result."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = FollowupJudgeResult(
            question_relevance=0.85,
            answer_grounding=0.65,
            diversity=0.75,
            explanation="Suggestions are relevant and diverse",
        )
        mock_chain_fn.return_value = mock_chain

        passed, qrel, agrnd, div, explanation = judge_followup_suggestions(
            question="What deals does Acme have?",
            suggestions=["Q1?", "Q2?", "Q3?"],
            answer="Acme has 3 deals.",
        )

        assert passed is True
        assert qrel == 0.85
        assert agrnd == 0.65
        assert div == 0.75
        assert "relevant" in explanation.lower()

    @patch("backend.eval.followup.judge.create_openai_chain")
    def test_judge_followup_failing(self, mock_chain_fn: MagicMock):
        """Test judge returns failing result."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = FollowupJudgeResult(
            question_relevance=0.4,
            answer_grounding=0.2,
            diversity=0.3,
            explanation="Suggestions lack variety",
        )
        mock_chain_fn.return_value = mock_chain

        passed, qrel, agrnd, div, explanation = judge_followup_suggestions(
            question="What deals does Acme have?",
            suggestions=["Q1?", "Q2?", "Q3?"],
            answer="Acme has 3 deals.",
        )

        assert passed is False
        assert qrel == 0.4
        assert agrnd == 0.2
        assert div == 0.3

    @patch("backend.eval.followup.judge.create_openai_chain")
    def test_judge_followup_error_propagates(self, mock_chain_fn: MagicMock):
        """Test judge lets exceptions propagate to caller."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = ValueError("API error")
        mock_chain_fn.return_value = mock_chain

        with pytest.raises(ValueError, match="API error"):
            judge_followup_suggestions(
                question="What deals does Acme have?",
                suggestions=["Q1?", "Q2?", "Q3?"],
            )

    @patch("backend.eval.followup.judge.create_openai_chain")
    def test_judge_followup_boundary(self, mock_chain_fn: MagicMock):
        """Test pass logic at SLO boundary values."""
        mock_chain = MagicMock()
        # Exactly at SLO thresholds: qrel=0.60, agrnd=0.50, div=0.50
        mock_chain.invoke.return_value = FollowupJudgeResult(
            question_relevance=0.6,
            answer_grounding=0.5,
            diversity=0.5,
            explanation="At threshold",
        )
        mock_chain_fn.return_value = mock_chain

        passed, qrel, agrnd, div, explanation = judge_followup_suggestions(
            question="Test?",
            suggestions=["Q1?", "Q2?", "Q3?"],
        )

        assert passed is True
        assert qrel == 0.6
        assert agrnd == 0.5
        assert div == 0.5

    @patch("backend.eval.followup.judge.create_openai_chain")
    def test_judge_fails_on_low_answer_grounding(self, mock_chain_fn: MagicMock):
        """Test that low answer_grounding alone causes failure."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = FollowupJudgeResult(
            question_relevance=0.9,
            answer_grounding=0.4,  # Below SLO of 0.50
            diversity=0.8,
            explanation="Generic follow-ups",
        )
        mock_chain_fn.return_value = mock_chain

        passed, _, _, _, _ = judge_followup_suggestions(
            question="Test?",
            suggestions=["Q1?", "Q2?", "Q3?"],
        )

        assert passed is False

    @patch("backend.eval.followup.judge.create_openai_chain")
    def test_judge_includes_answer_in_prompt(self, mock_chain_fn: MagicMock):
        """Test judge passes answer to the chain prompt."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = FollowupJudgeResult(
            question_relevance=0.8,
            answer_grounding=0.6,
            diversity=0.7,
            explanation="Good",
        )
        mock_chain_fn.return_value = mock_chain

        judge_followup_suggestions(
            question="What deals?",
            suggestions=["Q1?"],
            answer="Acme has 3 deals.",
        )

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert call_kwargs["answer"] == "Acme has 3 deals."

    @patch("backend.eval.followup.judge.create_openai_chain")
    def test_judge_empty_answer(self, mock_chain_fn: MagicMock):
        """Test judge with no answer passes empty answer."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = FollowupJudgeResult(
            question_relevance=0.8,
            answer_grounding=0.6,
            diversity=0.7,
            explanation="Good",
        )
        mock_chain_fn.return_value = mock_chain

        judge_followup_suggestions(
            question="What deals?",
            suggestions=["Q1?"],
        )

        call_kwargs = mock_chain.invoke.call_args[0][0]
        assert call_kwargs["answer"] == ""
