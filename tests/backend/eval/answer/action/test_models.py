"""Tests for backend.eval.answer.action.models module."""

from __future__ import annotations

from backend.eval.answer.action.models import ActionCaseResult, ActionEvalResults


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
        assert case.errors == []

    def test_action_case_result_with_scores(self):
        """Test ActionCaseResult with judge scores."""
        case = ActionCaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action="Send email",
            relevance=0.8,
            actionability=0.9,
            appropriateness=0.85,
            action_passed=True,
        )
        assert case.relevance == 0.8
        assert case.actionability == 0.9
        assert case.appropriateness == 0.85
        assert case.action_passed is True

    def test_action_case_result_passed_success(self):
        """Test passed property when action passes."""
        case = ActionCaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action="Send email",
            action_passed=True,
        )
        assert case.passed is True

    def test_action_case_result_passed_failure(self):
        """Test passed property when action fails."""
        case = ActionCaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action="Send email",
            action_passed=False,
        )
        assert case.passed is False

    def test_action_case_result_passed_no_action(self):
        """Test passed property when no action (passes by default)."""
        case = ActionCaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action=None,
            action_passed=True,
        )
        assert case.passed is True

    def test_action_case_result_passed_with_errors(self):
        """Test passed property is False when errors present."""
        case = ActionCaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action="Send email",
            action_passed=True,
            errors=["SQL error"],
        )
        assert case.passed is False


class TestActionEvalResults:
    """Tests for ActionEvalResults dataclass."""

    def test_action_eval_results_defaults(self):
        """Test ActionEvalResults default values."""
        results = ActionEvalResults()
        assert results.total == 0
        assert results.passed == 0
        assert results.cases == []
        assert results.total_with_actions == 0

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

    def test_action_eval_results_action_pass_rate(self):
        """Test action_pass_rate property calculation."""
        results = ActionEvalResults(total_with_actions=5)
        results.cases = [
            ActionCaseResult(
                question="Q1",
                answer="A1",
                suggested_action="Action 1",
                action_passed=True,
            ),
            ActionCaseResult(
                question="Q2",
                answer="A2",
                suggested_action="Action 2",
                action_passed=True,
            ),
            ActionCaseResult(
                question="Q3",
                answer="A3",
                suggested_action="Action 3",
                action_passed=False,
            ),
            ActionCaseResult(
                question="Q4",
                answer="A4",
                suggested_action=None,
                action_passed=True,
            ),
        ]
        # 2 passed out of 3 with actions (Q4 has no action)
        # Actually total_with_actions=5 is wrong for this test, let's fix
        results.total_with_actions = 3
        assert results.action_pass_rate == 2 / 3

    def test_action_eval_results_action_pass_rate_zero(self):
        """Test action_pass_rate with no actions."""
        results = ActionEvalResults(total_with_actions=0)
        assert results.action_pass_rate == 0.0

    def test_action_eval_results_compute_aggregates_empty(self):
        """Test compute_aggregates with empty cases."""
        results = ActionEvalResults()
        results.compute_aggregates()
        assert results.avg_relevance == 0.0

    def test_action_eval_results_compute_aggregates(self):
        """Test compute_aggregates computes averages."""
        results = ActionEvalResults()
        results.cases = [
            ActionCaseResult(
                question="Q1",
                answer="A1",
                suggested_action="Action 1",
                relevance=0.7,
                actionability=0.8,
                appropriateness=0.9,
            ),
            ActionCaseResult(
                question="Q2",
                answer="A2",
                suggested_action="Action 2",
                relevance=0.9,
                actionability=0.85,
                appropriateness=0.95,
            ),
            ActionCaseResult(
                question="Q3",
                answer="A3",
                suggested_action=None,  # No action
            ),
        ]
        results.compute_aggregates()

        # Action metrics only from cases with actions (Q1, Q2)
        assert results.avg_relevance == 0.8  # (0.7 + 0.9) / 2
        assert results.avg_actionability == 0.825  # (0.8 + 0.85) / 2
        assert results.avg_appropriateness == 0.925  # (0.9 + 0.95) / 2
