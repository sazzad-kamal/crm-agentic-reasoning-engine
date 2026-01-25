"""Tests for backend.eval.answer.text.models module."""

from __future__ import annotations

from backend.eval.answer.text.models import TextCaseResult, TextEvalResults


class TestTextCaseResult:
    """Tests for TextCaseResult dataclass."""

    def test_text_case_result_basic(self):
        """Test basic TextCaseResult creation."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
        )
        assert case.question == "Test question"
        assert case.answer == "Test answer"
        assert case.errors == []

    def test_text_case_result_with_scores(self):
        """Test TextCaseResult with RAGAS scores."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
            faithfulness_score=0.8,
            relevance_score=0.9,
            answer_correctness_score=0.85,
        )
        assert case.faithfulness_score == 0.8
        assert case.relevance_score == 0.9
        assert case.answer_correctness_score == 0.85

    def test_text_case_result_passed_success(self):
        """Test passed property when scores are good."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
            faithfulness_score=0.8,
            relevance_score=0.9,
        )
        assert case.passed is True

    def test_text_case_result_passed_failure_faithfulness(self):
        """Test passed property when faithfulness is low."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
            faithfulness_score=0.5,  # Below 0.6 threshold
            relevance_score=0.9,
        )
        assert case.passed is False

    def test_text_case_result_passed_failure_relevance(self):
        """Test passed property when relevance is low."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
            faithfulness_score=0.8,
            relevance_score=0.5,  # Below 0.6 threshold
        )
        assert case.passed is False

    def test_text_case_result_passed_with_errors(self):
        """Test passed property is False when errors present."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
            faithfulness_score=0.8,
            relevance_score=0.9,
            errors=["SQL error"],
        )
        assert case.passed is False


class TestTextEvalResults:
    """Tests for TextEvalResults dataclass."""

    def test_text_eval_results_defaults(self):
        """Test TextEvalResults default values."""
        results = TextEvalResults()
        assert results.total == 0
        assert results.passed == 0
        assert results.cases == []

    def test_text_eval_results_failed_property(self):
        """Test failed property calculation."""
        results = TextEvalResults(total=10, passed=7)
        assert results.failed == 3

    def test_text_eval_results_pass_rate(self):
        """Test pass_rate property calculation."""
        results = TextEvalResults(total=10, passed=8)
        assert results.pass_rate == 0.8

    def test_text_eval_results_pass_rate_zero_total(self):
        """Test pass_rate with zero total."""
        results = TextEvalResults(total=0, passed=0)
        assert results.pass_rate == 0.0

    def test_text_eval_results_compute_aggregates_empty(self):
        """Test compute_aggregates with empty cases."""
        results = TextEvalResults()
        results.compute_aggregates()
        assert results.avg_faithfulness == 0.0

    def test_text_eval_results_compute_aggregates(self):
        """Test compute_aggregates computes averages."""
        results = TextEvalResults()
        results.cases = [
            TextCaseResult(
                question="Q1",
                answer="A1",
                faithfulness_score=0.7,
                relevance_score=0.8,
                answer_correctness_score=0.75,
            ),
            TextCaseResult(
                question="Q2",
                answer="A2",
                faithfulness_score=0.9,
                relevance_score=0.85,
                answer_correctness_score=0.95,
            ),
        ]
        results.compute_aggregates()

        assert results.avg_faithfulness == 0.8
        assert results.avg_relevance == 0.825
        assert results.avg_answer_correctness == 0.85
