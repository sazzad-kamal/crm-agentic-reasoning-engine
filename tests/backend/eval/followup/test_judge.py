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
            relevance=0.8,
            diversity=0.7,
            explanation="Good suggestions",
        )
        assert result.relevance == 0.8
        assert result.diversity == 0.7
        assert result.explanation == "Good suggestions"


class TestJudgeFollowupSuggestions:
    """Tests for judge_followup_suggestions function."""

    @patch("backend.eval.followup.judge.create_openai_chain")
    def test_judge_followup_passing(self, mock_chain_fn: MagicMock):
        """Test judge returns passing result."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = FollowupJudgeResult(
            relevance=0.85,
            diversity=0.75,
            explanation="Suggestions are relevant and diverse",
        )
        mock_chain_fn.return_value = mock_chain

        passed, rel, div, explanation = judge_followup_suggestions(
            question="What deals does Acme have?",
            suggestions=["Q1?", "Q2?", "Q3?"],
        )

        assert passed is True
        assert rel == 0.85
        assert div == 0.75
        assert "relevant" in explanation.lower()

    @patch("backend.eval.followup.judge.create_openai_chain")
    def test_judge_followup_failing(self, mock_chain_fn: MagicMock):
        """Test judge returns failing result."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = FollowupJudgeResult(
            relevance=0.4,
            diversity=0.3,
            explanation="Suggestions lack variety",
        )
        mock_chain_fn.return_value = mock_chain

        passed, rel, div, explanation = judge_followup_suggestions(
            question="What deals does Acme have?",
            suggestions=["Q1?", "Q2?", "Q3?"],
        )

        assert passed is False
        assert rel == 0.4
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
    def test_judge_followup_relevance_boundary(self, mock_chain_fn: MagicMock):
        """Test pass logic at SLO boundary values."""
        mock_chain = MagicMock()
        # Exactly at SLO thresholds: relevance=0.60, diversity=0.50
        mock_chain.invoke.return_value = FollowupJudgeResult(
            relevance=0.6,
            diversity=0.5,
            explanation="At threshold",
        )
        mock_chain_fn.return_value = mock_chain

        passed, rel, div, explanation = judge_followup_suggestions(
            question="Test?",
            suggestions=["Q1?", "Q2?", "Q3?"],
        )

        assert passed is True
        assert rel == 0.6
        assert div == 0.5
