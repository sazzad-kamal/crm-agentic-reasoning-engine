"""Tests for backend.eval.answer.action.models module."""

from __future__ import annotations

from backend.eval.answer.action.models import (
    SLO_ACTION_PASS_RATE,
    SLO_ACTIONABILITY,
    SLO_APPROPRIATENESS,
    SLO_RELEVANCE,
    ActionCaseResult,
    ActionEvalResults,
)


class TestSLOConstants:
    """Tests for SLO constants."""

    def test_slo_action_pass_rate(self):
        """Test SLO_ACTION_PASS_RATE value."""
        assert SLO_ACTION_PASS_RATE == 0.80

    def test_slo_relevance(self):
        """Test SLO_RELEVANCE value."""
        assert SLO_RELEVANCE == 0.90

    def test_slo_actionability(self):
        """Test SLO_ACTIONABILITY value."""
        assert SLO_ACTIONABILITY == 0.70

    def test_slo_appropriateness(self):
        """Test SLO_APPROPRIATENESS value."""
        assert SLO_APPROPRIATENESS == 0.85


class TestActionCaseResult:
    """Tests for ActionCaseResult dataclass."""

    def test_action_case_result_basic(self):
        """Test basic ActionCaseResult creation."""
        case = ActionCaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action="Send email",
        )
        assert case.question == "Test question"
        assert case.answer == "Test answer"
        assert case.suggested_action == "Send email"
        assert case.expected_action is False
        assert case.errors == []

    def test_action_case_result_with_scores(self):
        """Test ActionCaseResult with judge scores."""
        case = ActionCaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action="Send email",
            expected_action=True,
            relevance=0.8,
            actionability=0.9,
            appropriateness=0.85,
            action_passed=True,
        )
        assert case.relevance == 0.8
        assert case.actionability == 0.9
        assert case.appropriateness == 0.85
        assert case.action_passed is True

    def test_action_expected_correct(self):
        """Test outcome: action expected, produced, judged pass."""
        case = ActionCaseResult(
            question="Q",
            answer="A",
            suggested_action="Send email",
            expected_action=True,
            action_passed=True,
        )
        assert case.action_passed is True

    def test_action_expected_failed(self):
        """Test outcome: action expected, produced, judged fail."""
        case = ActionCaseResult(
            question="Q",
            answer="A",
            suggested_action="Send email",
            expected_action=True,
            action_passed=False,
        )
        assert case.action_passed is False

    def test_action_missing(self):
        """Test outcome: action expected but not produced."""
        case = ActionCaseResult(
            question="Q",
            answer="A",
            suggested_action=None,
            expected_action=True,
            action_passed=False,
        )
        assert case.action_passed is False

    def test_spurious_action(self):
        """Test outcome: action not expected but produced."""
        case = ActionCaseResult(
            question="Q",
            answer="A",
            suggested_action="Unwanted action",
            expected_action=False,
            action_passed=False,
        )
        assert case.action_passed is False

    def test_correct_silence(self):
        """Test outcome: action not expected and not produced."""
        case = ActionCaseResult(
            question="Q",
            answer="A",
            suggested_action=None,
            expected_action=False,
            action_passed=True,
        )
        assert case.action_passed is True


class TestActionEvalResults:
    """Tests for ActionEvalResults dataclass."""

    def test_action_eval_results_defaults(self):
        """Test ActionEvalResults default values."""
        results = ActionEvalResults()
        assert results.total == 0
        assert results.passed == 0
        assert results.cases == []
        assert results.action_expected_passed == 0
        assert results.action_expected_failed == 0
        assert results.action_missing == 0
        assert results.spurious_action == 0
        assert results.correct_silence == 0
        assert results.error_count == 0

    def test_action_eval_results_failed_property(self):
        """Test failed property calculation."""
        results = ActionEvalResults(total=10, passed=7)
        assert results.failed == 3

    def test_action_eval_results_pass_rate(self):
        """Test pass_rate property calculation."""
        results = ActionEvalResults(total=10, passed=8)
        assert results.pass_rate == 0.8

    def test_action_eval_results_pass_rate_zero_total(self):
        """Test pass_rate with zero total."""
        results = ActionEvalResults(total=0, passed=0)
        assert results.pass_rate == 0.0

    def test_action_eval_results_compute_aggregates_empty(self):
        """Test compute_aggregates with empty cases."""
        results = ActionEvalResults()
        results.compute_aggregates()
        assert results.avg_relevance == 0.0

    def test_action_eval_results_compute_aggregates(self):
        """Test compute_aggregates computes averages, passed, and breakdown."""
        results = ActionEvalResults(total=5)
        results.cases = [
            # Outcome 1: action expected + correct (judged pass)
            ActionCaseResult(
                question="Q1",
                answer="A1",
                suggested_action="Action 1",
                expected_action=True,
                relevance=0.7,
                actionability=0.8,
                appropriateness=0.9,
                action_passed=True,
            ),
            # Outcome 2: action expected + failed (judged fail)
            ActionCaseResult(
                question="Q2",
                answer="A2",
                suggested_action="Action 2",
                expected_action=True,
                relevance=0.9,
                actionability=0.85,
                appropriateness=0.95,
                action_passed=False,
            ),
            # Outcome 3: action missing
            ActionCaseResult(
                question="Q3",
                answer="A3",
                suggested_action=None,
                expected_action=True,
                action_passed=False,
            ),
            # Outcome 4: spurious action
            ActionCaseResult(
                question="Q4",
                answer="A4",
                suggested_action="Unwanted",
                expected_action=False,
                action_passed=False,
            ),
            # Outcome 5: correct silence
            ActionCaseResult(
                question="Q5",
                answer="A5",
                suggested_action=None,
                expected_action=False,
                action_passed=True,
            ),
        ]
        results.compute_aggregates()

        assert results.passed == 2  # Q1 + Q5

        # Breakdown
        assert results.action_expected_passed == 1  # Q1
        assert results.action_expected_failed == 1  # Q2
        assert results.action_missing == 1  # Q3
        assert results.spurious_action == 1  # Q4
        assert results.correct_silence == 1  # Q5

        # Action metrics only from judged cases (expected + produced): Q1, Q2
        assert results.avg_relevance == 0.8  # (0.7 + 0.9) / 2
        assert results.avg_actionability == 0.825  # (0.8 + 0.85) / 2
        assert results.avg_appropriateness == 0.925  # (0.9 + 0.95) / 2

    def test_compute_aggregates_error_cases_excluded_from_breakdown(self):
        """Test that error cases are excluded from breakdown counts."""
        results = ActionEvalResults(total=2)
        results.cases = [
            ActionCaseResult(
                question="Q1",
                answer="",
                suggested_action=None,
                expected_action=True,
                errors=["SQL error"],
            ),
            ActionCaseResult(
                question="Q2",
                answer="A2",
                suggested_action=None,
                expected_action=False,
                action_passed=True,
            ),
        ]
        results.compute_aggregates()

        assert results.passed == 1  # Only Q2
        assert results.error_count == 1  # Q1 has errors
        assert results.action_missing == 0  # Q1 excluded from breakdown
        assert results.correct_silence == 1  # Q2

    def test_compute_aggregates_no_judged_cases(self):
        """Test metrics stay zero when no cases were judged."""
        results = ActionEvalResults(total=1)
        results.cases = [
            ActionCaseResult(
                question="Q1",
                answer="A1",
                suggested_action=None,
                expected_action=False,
                action_passed=True,
            ),
        ]
        results.compute_aggregates()

        assert results.avg_relevance == 0.0
        assert results.avg_actionability == 0.0
        assert results.avg_appropriateness == 0.0
