"""Tests for backend.eval.followup.models module."""

from __future__ import annotations

import pytest

from backend.eval.followup.models import (
    SLO_FOLLOWUP_ANSWER_GROUNDING,
    SLO_FOLLOWUP_DIVERSITY,
    SLO_FOLLOWUP_PASS_RATE,
    SLO_FOLLOWUP_QUESTION_RELEVANCE,
    FollowupCaseResult,
    FollowupEvalResults,
)


class TestSLOConstants:
    """Tests for SLO constants."""

    def test_slo_followup_pass_rate(self):
        """Test SLO_FOLLOWUP_PASS_RATE value."""
        assert SLO_FOLLOWUP_PASS_RATE == 0.80

    def test_slo_followup_question_relevance(self):
        """Test SLO_FOLLOWUP_QUESTION_RELEVANCE value."""
        assert SLO_FOLLOWUP_QUESTION_RELEVANCE == 0.60

    def test_slo_followup_answer_grounding(self):
        """Test SLO_FOLLOWUP_ANSWER_GROUNDING value."""
        assert SLO_FOLLOWUP_ANSWER_GROUNDING == 0.50

    def test_slo_followup_diversity(self):
        """Test SLO_FOLLOWUP_DIVERSITY value."""
        assert SLO_FOLLOWUP_DIVERSITY == 0.50


class TestFollowupCaseResult:
    """Tests for FollowupCaseResult model."""

    def test_followup_case_result_basic(self):
        """Test basic FollowupCaseResult creation."""
        case = FollowupCaseResult(question="Test question")
        assert case.question == "Test question"
        assert case.answer == ""
        assert case.suggestions == []
        assert case.passed is False
        assert case.question_relevance == 0.0
        assert case.answer_grounding == 0.0
        assert case.diversity == 0.0
        assert case.explanation == ""
        assert case.errors == []

    def test_followup_case_result_with_answer(self):
        """Test FollowupCaseResult with answer field."""
        case = FollowupCaseResult(
            question="What deals does Acme have?",
            answer="Acme has 3 deals.",
        )
        assert case.answer == "Acme has 3 deals."

    def test_followup_case_result_with_scores(self):
        """Test FollowupCaseResult with judge scores."""
        case = FollowupCaseResult(
            question="Test question",
            suggestions=["Q1?", "Q2?", "Q3?"],
            passed=True,
            question_relevance=0.8,
            answer_grounding=0.6,
            diversity=0.7,
            explanation="Good suggestions",
        )
        assert case.question_relevance == 0.8
        assert case.answer_grounding == 0.6
        assert case.diversity == 0.7
        assert case.passed is True
        assert case.explanation == "Good suggestions"
        assert len(case.suggestions) == 3

    def test_followup_case_result_with_errors(self):
        """Test FollowupCaseResult with errors."""
        case = FollowupCaseResult(
            question="Test question",
            errors=["Generation error: timeout"],
        )
        assert case.passed is False
        assert case.errors == ["Generation error: timeout"]


class TestFollowupEvalResults:
    """Tests for FollowupEvalResults model."""

    def test_followup_eval_results_defaults(self):
        """Test FollowupEvalResults default values."""
        results = FollowupEvalResults()
        assert results.total == 0
        assert results.passed == 0
        assert results.cases == []
        assert results.avg_question_relevance == 0.0
        assert results.avg_answer_grounding == 0.0
        assert results.avg_diversity == 0.0

    def test_followup_eval_results_failed_property(self):
        """Test failed property calculation."""
        results = FollowupEvalResults(total=10, passed=7)
        assert results.failed == 3

    def test_followup_eval_results_pass_rate(self):
        """Test pass_rate property calculation."""
        results = FollowupEvalResults(total=10, passed=8)
        assert results.pass_rate == 0.8

    def test_followup_eval_results_pass_rate_zero_total(self):
        """Test pass_rate with zero total."""
        results = FollowupEvalResults(total=0, passed=0)
        assert results.pass_rate == 0.0

    def test_followup_eval_results_compute_aggregates_empty(self):
        """Test compute_aggregates with empty cases."""
        results = FollowupEvalResults()
        results.compute_aggregates()
        assert results.avg_question_relevance == 0.0
        assert results.avg_answer_grounding == 0.0
        assert results.avg_diversity == 0.0

    def test_followup_eval_results_compute_aggregates(self):
        """Test compute_aggregates computes passed count and averages."""
        results = FollowupEvalResults(total=3)
        results.cases = [
            FollowupCaseResult(
                question="Q1",
                suggestions=["A", "B", "C"],
                passed=True,
                question_relevance=0.8,
                answer_grounding=0.6,
                diversity=0.7,
            ),
            FollowupCaseResult(
                question="Q2",
                suggestions=["A", "B", "C"],
                passed=False,
                question_relevance=0.4,
                answer_grounding=0.2,
                diversity=0.3,
            ),
            FollowupCaseResult(
                question="Q3",
                suggestions=["A", "B", "C"],
                passed=True,
                question_relevance=0.9,
                answer_grounding=0.7,
                diversity=0.8,
            ),
        ]
        results.compute_aggregates()

        assert results.passed == 2
        assert results.avg_question_relevance == pytest.approx(0.7)  # (0.8 + 0.4 + 0.9) / 3
        assert results.avg_answer_grounding == pytest.approx(0.5)  # (0.6 + 0.2 + 0.7) / 3
        assert results.avg_diversity == pytest.approx(0.6)  # (0.7 + 0.3 + 0.8) / 3

    def test_compute_aggregates_with_error_cases(self):
        """Test compute_aggregates includes error cases in averages."""
        results = FollowupEvalResults(total=2)
        results.cases = [
            FollowupCaseResult(
                question="Q1",
                passed=True,
                question_relevance=0.8,
                answer_grounding=0.6,
                diversity=0.6,
            ),
            FollowupCaseResult(
                question="Q2",
                errors=["Generation error"],
                question_relevance=0.0,
                answer_grounding=0.0,
                diversity=0.0,
            ),
        ]
        results.compute_aggregates()

        assert results.passed == 1
        assert results.avg_question_relevance == 0.4  # (0.8 + 0.0) / 2
        assert results.avg_answer_grounding == 0.3  # (0.6 + 0.0) / 2
        assert results.avg_diversity == 0.3  # (0.6 + 0.0) / 2
