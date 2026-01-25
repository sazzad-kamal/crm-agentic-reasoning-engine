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
        """Test TextCaseResult with answer correctness score."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
            answer_correctness_score=0.85,
            answer_relevancy_score=0.90,
        )
        assert case.answer_correctness_score == 0.85
        assert case.answer_relevancy_score == 0.90

    def test_text_case_result_passed_success(self):
        """Test passed property when both scores are good."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
            answer_correctness_score=0.55,  # >= 0.50 threshold
            answer_relevancy_score=0.90,  # >= 0.85 threshold
        )
        assert case.passed is True

    def test_text_case_result_passed_failure_low_correctness(self):
        """Test passed property when answer_correctness is low."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
            answer_correctness_score=0.45,  # Below 0.50 threshold
            answer_relevancy_score=0.90,
        )
        assert case.passed is False

    def test_text_case_result_passed_failure_low_relevancy(self):
        """Test passed property when answer_relevancy is low."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
            answer_correctness_score=0.55,
            answer_relevancy_score=0.80,  # Below 0.85 threshold
        )
        assert case.passed is False

    def test_text_case_result_passed_with_errors(self):
        """Test passed property is False when errors present."""
        case = TextCaseResult(
            question="Test question",
            answer="Test answer",
            answer_correctness_score=0.55,
            answer_relevancy_score=0.90,
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
        assert results.avg_answer_correctness == 0.0

    def test_text_eval_results_compute_aggregates(self):
        """Test compute_aggregates computes averages."""
        results = TextEvalResults()
        results.cases = [
            TextCaseResult(
                question="Q1",
                answer="A1",
                answer_correctness_score=0.75,
                answer_relevancy_score=0.90,
            ),
            TextCaseResult(
                question="Q2",
                answer="A2",
                answer_correctness_score=0.95,
                answer_relevancy_score=0.92,
            ),
        ]
        results.compute_aggregates()

        assert results.avg_answer_correctness == 0.85
        assert results.avg_answer_relevancy == 0.91

    def test_text_eval_results_ragas_success_rate_with_metrics(self):
        """Test ragas_success_rate when metrics are present."""
        results = TextEvalResults(
            ragas_metrics_total=10,
            ragas_metrics_failed=2,
        )
        assert results.ragas_success_rate == 0.8

    def test_text_eval_results_ragas_success_rate_zero_total(self):
        """Test ragas_success_rate returns 1.0 when no metrics."""
        results = TextEvalResults(ragas_metrics_total=0)
        assert results.ragas_success_rate == 1.0
