"""Tests for 5-dimension answer text quality judge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.eval.answer.text.judge import TextJudgeResult, judge_answer_text


class TestTextJudgeResult:
    """Tests for TextJudgeResult model."""

    def test_creation(self):
        result = TextJudgeResult(
            grounding=0.9,
            completeness=0.8,
            clarity=0.85,
            accuracy=0.95,
            actionability=0.7,
            explanation="Good answer",
        )
        assert result.grounding == 0.9
        assert result.completeness == 0.8
        assert result.clarity == 0.85
        assert result.accuracy == 0.95
        assert result.actionability == 0.7


class TestJudgeAnswerText:
    """Tests for judge_answer_text function."""

    def test_passing_scores(self):
        mock_result = TextJudgeResult(
            grounding=0.9,
            completeness=0.8,
            clarity=0.85,
            accuracy=0.9,
            actionability=0.7,
            explanation="Well grounded and complete",
        )
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result

        with patch("backend.eval.answer.text.judge.create_openai_chain", return_value=mock_chain):
            passed, gr, co, cl, ac, act, explanation = judge_answer_text(
                question="What deals are closing?",
                answer="There are 3 deals closing [E1].",
                context='{"data": []}',
            )

        assert passed is True
        assert gr == 0.9
        assert co == 0.8
        assert cl == 0.85
        assert ac == 0.9
        assert act == 0.7
        assert "grounded" in explanation.lower()

    def test_failing_scores(self):
        mock_result = TextJudgeResult(
            grounding=0.3,
            completeness=0.4,
            clarity=0.5,
            accuracy=0.3,
            actionability=0.2,
            explanation="Poor quality",
        )
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result

        with patch("backend.eval.answer.text.judge.create_openai_chain", return_value=mock_chain):
            passed, gr, co, cl, ac, act, explanation = judge_answer_text(
                question="Q", answer="A", context="{}",
            )

        assert passed is False
        assert gr == 0.3

    def test_error_propagates(self):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API error")

        with patch("backend.eval.answer.text.judge.create_openai_chain", return_value=mock_chain):
            with pytest.raises(Exception, match="API error"):
                judge_answer_text(question="Q", answer="A", context="{}")

    def test_partial_pass_fails(self):
        """One dimension below SLO should fail overall."""
        mock_result = TextJudgeResult(
            grounding=0.9,
            completeness=0.9,
            clarity=0.9,
            accuracy=0.5,  # Below SLO_JUDGE_ACCURACY (0.80)
            actionability=0.9,
            explanation="Accuracy issue",
        )
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result

        with patch("backend.eval.answer.text.judge.create_openai_chain", return_value=mock_chain):
            passed, _, _, _, _, _, _ = judge_answer_text(
                question="Q", answer="A", context="{}",
            )

        assert passed is False
