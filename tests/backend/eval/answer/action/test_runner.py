"""Tests for backend.eval.answer.action.runner module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.eval.answer.action.models import ActionCaseResult, ActionEvalResults
from backend.eval.answer.action.runner import print_summary, run_action_eval
from backend.eval.answer.shared.models import Question


class TestRunActionEval:
    """Tests for run_action_eval function."""

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.judge_suggested_action")
    def test_run_action_eval_with_action(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test action evaluation with suggested action."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ("Answer 1", "Send email to client", [{}], None)
        mock_judge.return_value = (True, 0.8, 0.9, 0.85, "Good action")

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 1
        assert results.total_with_actions == 1
        assert results.cases[0].suggested_action == "Send email to client"
        assert results.cases[0].relevance == 0.8

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    def test_run_action_eval_no_action(
        self,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test action evaluation without suggested action."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ("Answer 1", None, [{}], None)

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 1  # Passes by default when no action
        assert results.total_with_actions == 0
        assert results.cases[0].suggested_action is None
        assert results.cases[0].action_passed is True

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    def test_run_action_eval_with_error(
        self,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test action evaluation with error."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ("", None, [], "SQL error: timeout")

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].errors == ["SQL error: timeout"]

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.judge_suggested_action")
    def test_run_action_eval_with_limit(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test action evaluation with limit parameter."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
            Question(text="Q2", expected_sql="SELECT 2"),
            Question(text="Q3", expected_sql="SELECT 3"),
        ]
        mock_generate.return_value = ("Answer", "Action", [{}], None)
        mock_judge.return_value = (True, 0.8, 0.9, 0.85, "Good")

        results = run_action_eval(limit=2)

        assert results.total == 2
        assert mock_generate.call_count == 2

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.judge_suggested_action")
    def test_run_action_eval_computes_aggregates(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test that aggregates are computed."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
            Question(text="Q2", expected_sql="SELECT 2"),
        ]
        mock_generate.side_effect = [
            ("Answer 1", "Action 1", [{}], None),
            ("Answer 2", "Action 2", [{}], None),
        ]
        mock_judge.side_effect = [
            (True, 0.7, 0.8, 0.9, "Good"),
            (True, 0.9, 0.85, 0.95, "Excellent"),
        ]

        results = run_action_eval()

        assert results.avg_relevance == 0.8  # (0.7 + 0.9) / 2


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary_passing(self, capsys):
        """Test print_summary with passing results."""
        results = ActionEvalResults(total=10, passed=9, total_with_actions=5)
        results.avg_relevance = 0.85
        results.avg_actionability = 0.90
        results.avg_appropriateness = 0.88
        results.cases = [
            ActionCaseResult(
                question="Q",
                answer="A",
                suggested_action="Action",
                action_passed=True,
            )
            for _ in range(5)
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "90.0%" in captured.out

    def test_print_summary_failing(self, capsys):
        """Test print_summary with failing results."""
        results = ActionEvalResults(total=10, passed=5, total_with_actions=3)
        results.cases = [
            ActionCaseResult(
                question="Failed question",
                answer="Answer",
                suggested_action="Bad action",
                relevance=0.3,
                actionability=0.4,
                appropriateness=0.5,
                action_passed=False,
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "Failed Cases" in captured.out

    def test_print_summary_no_actions(self, capsys):
        """Test print_summary with no action cases."""
        results = ActionEvalResults(total=5, passed=5, total_with_actions=0)

        print_summary(results)

        captured = capsys.readouterr()
        assert "0 cases with actions" in captured.out
