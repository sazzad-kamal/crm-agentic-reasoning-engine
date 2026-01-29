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
    @patch("backend.eval.answer.action.runner.generate_action")
    @patch("backend.eval.answer.action.runner.judge_suggested_action")
    def test_action_expected_and_correct(
        self,
        mock_judge: MagicMock,
        mock_action: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test outcome: action expected, produced, judged pass."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1", expected_action=True),
        ]
        mock_generate.return_value = ("Answer 1", [{}], None)
        mock_action.return_value = ("Send email to client", None)
        mock_judge.return_value = (True, 0.8, 0.9, 0.85, "Good action")

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 1
        assert results.action_expected_passed == 1
        assert results.cases[0].suggested_action == "Send email to client"
        assert results.cases[0].expected_action is True
        assert results.cases[0].relevance == 0.8
        assert results.cases[0].explanation == "Good action"
        mock_judge.assert_called_once()

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.generate_action")
    @patch("backend.eval.answer.action.runner.judge_suggested_action")
    def test_action_expected_and_failed(
        self,
        mock_judge: MagicMock,
        mock_action: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test outcome: action expected, produced, judged fail."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1", expected_action=True),
        ]
        mock_generate.return_value = ("Answer", [{}], None)
        mock_action.return_value = ("Bad action", None)
        mock_judge.return_value = (False, 0.3, 0.2, 0.4, "Poor action")

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.action_expected_failed == 1
        assert results.cases[0].action_passed is False
        assert results.cases[0].explanation == "Poor action"
        mock_judge.assert_called_once()

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.generate_action")
    def test_action_missing(
        self,
        mock_action: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test outcome: action expected but not produced."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1", expected_action=True),
        ]
        mock_generate.return_value = ("Answer 1", [{}], None)
        mock_action.return_value = (None, None)

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.action_missing == 1
        assert results.cases[0].expected_action is True
        assert results.cases[0].suggested_action is None
        assert results.cases[0].action_passed is False

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.generate_action")
    @patch("backend.eval.answer.action.runner.judge_suggested_action")
    def test_spurious_action(
        self,
        mock_judge: MagicMock,
        mock_action: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test outcome: action not expected but produced (spurious)."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1", expected_action=False),
        ]
        mock_generate.return_value = ("Answer 1", [{}], None)
        mock_action.return_value = ("Unwanted action", None)

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.spurious_action == 1
        assert results.cases[0].expected_action is False
        assert results.cases[0].suggested_action == "Unwanted action"
        assert results.cases[0].action_passed is False
        mock_judge.assert_not_called()

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.generate_action")
    def test_correct_silence(
        self,
        mock_action: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test outcome: action not expected and not produced."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1", expected_action=False),
        ]
        mock_generate.return_value = ("Answer 1", [{}], None)
        mock_action.return_value = (None, None)

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 1
        assert results.correct_silence == 1
        assert results.cases[0].expected_action is False
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
            Question(text="Q1", expected_sql="SELECT 1", expected_action=True),
        ]
        mock_generate.return_value = ("", [], "SQL error: timeout")

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].errors == ["SQL error: timeout"]
        assert results.cases[0].expected_action is True

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.generate_action")
    def test_run_action_eval_with_action_error(
        self,
        mock_action: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test action evaluation when generate_action returns error."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1", expected_action=True),
        ]
        mock_generate.return_value = ("Answer", [{}], None)
        mock_action.return_value = (None, "Action error: Chain failed")

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].errors == ["Action error: Chain failed"]

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.generate_action")
    @patch("backend.eval.answer.action.runner.judge_suggested_action")
    def test_run_action_eval_judge_exception(
        self,
        mock_judge: MagicMock,
        mock_action: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test action evaluation when judge raises exception."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1", expected_action=True),
        ]
        mock_generate.return_value = ("Answer", [{}], None)
        mock_action.return_value = ("Some action", None)
        mock_judge.side_effect = ValueError("Judge internal error")

        results = run_action_eval()

        assert results.total == 1
        assert results.passed == 0
        assert "Judge failed: Judge internal error" in results.cases[0].errors[0]

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.generate_action")
    @patch("backend.eval.answer.action.runner.judge_suggested_action")
    def test_run_action_eval_with_limit(
        self,
        mock_judge: MagicMock,
        mock_action: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test action evaluation with limit parameter."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1", expected_action=True),
            Question(text="Q2", expected_sql="SELECT 2", expected_action=True),
            Question(text="Q3", expected_sql="SELECT 3", expected_action=True),
        ]
        mock_generate.return_value = ("Answer", [{}], None)
        mock_action.return_value = ("Action", None)
        mock_judge.return_value = (True, 0.8, 0.9, 0.85, "Good")

        results = run_action_eval(limit=2)

        assert results.total == 2
        assert mock_generate.call_count == 2

    @patch("backend.eval.answer.action.runner.get_connection")
    @patch("backend.eval.answer.action.runner.load_questions")
    @patch("backend.eval.answer.action.runner.generate_answer")
    @patch("backend.eval.answer.action.runner.generate_action")
    @patch("backend.eval.answer.action.runner.judge_suggested_action")
    def test_run_action_eval_computes_aggregates(
        self,
        mock_judge: MagicMock,
        mock_action: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test that aggregates are computed including breakdown."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1", expected_action=True),
            Question(text="Q2", expected_sql="SELECT 2", expected_action=False),
        ]
        mock_generate.side_effect = [
            ("Answer 1", [{}], None),
            ("Answer 2", [{}], None),
        ]
        mock_action.side_effect = [
            ("Action 1", None),
            (None, None),
        ]
        mock_judge.return_value = (True, 0.7, 0.8, 0.9, "Good")

        results = run_action_eval()

        assert results.passed == 2
        assert results.action_expected_passed == 1
        assert results.correct_silence == 1
        assert results.avg_relevance == 0.7


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary_passing(self, capsys):
        """Test print_summary with passing results."""
        results = ActionEvalResults(
            total=10,
            passed=9,
            action_expected_passed=4,
            action_expected_failed=1,
            action_missing=0,
            spurious_action=0,
            correct_silence=5,
        )
        results.avg_relevance = 0.85
        results.avg_actionability = 0.90
        results.avg_appropriateness = 0.88
        results.cases = [
            ActionCaseResult(
                question="Q",
                answer="A",
                suggested_action="Action",
                expected_action=True,
                action_passed=True,
            )
            for _ in range(5)
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "90.0%" in captured.out
        assert ">=80.0% SLO" in captured.out
        assert "Action expected:      4 passed, 1 failed (judged), 0 missing" in captured.out
        assert "No action expected:   5 passed (quiet), 0 failed (spurious)" in captured.out

    def test_print_summary_failing(self, capsys):
        """Test print_summary with failing results."""
        results = ActionEvalResults(
            total=10,
            passed=5,
            action_expected_passed=1,
            action_expected_failed=2,
        )
        results.cases = [
            ActionCaseResult(
                question="Failed question",
                answer="Answer",
                suggested_action="Bad action",
                expected_action=True,
                relevance=0.3,
                actionability=0.4,
                appropriateness=0.5,
                action_passed=False,
                explanation="Action is too generic",
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "All Cases" in captured.out
        assert "Answer: Answer" in captured.out
        assert "Judge: Action is too generic" in captured.out

    def test_print_summary_shows_full_answer(self, capsys):
        """Test print_summary shows full answers."""
        results = ActionEvalResults(total=1, passed=0)
        results.cases = [
            ActionCaseResult(
                question="Q",
                answer="A" * 150,
                suggested_action="Action",
                expected_action=True,
                relevance=0.3,
                actionability=0.4,
                appropriateness=0.5,
                action_passed=False,
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "Answer: " + "A" * 150 in captured.out

    def test_print_summary_no_answer_line_when_empty(self, capsys):
        """Test print_summary omits answer line when answer is empty."""
        results = ActionEvalResults(
            total=1,
            passed=0,
            action_missing=1,
        )
        results.cases = [
            ActionCaseResult(
                question="Q",
                answer="",
                suggested_action=None,
                expected_action=True,
                action_passed=False,
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "Answer:" not in captured.out

    def test_print_summary_no_actions(self, capsys):
        """Test print_summary with no action cases."""
        results = ActionEvalResults(
            total=5,
            passed=5,
            correct_silence=5,
        )
        results.cases = []

        print_summary(results)

        captured = capsys.readouterr()
        assert "No action expected:   5 passed (quiet), 0 failed (spurious)" in captured.out

    def test_print_summary_with_error_case(self, capsys):
        """Test print_summary with a failed case that has errors."""
        results = ActionEvalResults(total=10, passed=5)
        results.cases = [
            ActionCaseResult(
                question="Question with error",
                answer="",
                suggested_action=None,
                expected_action=True,
                errors=["SQL timeout", "Connection failed"],
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "All Cases (1)" in captured.out
        assert "Error: SQL timeout; Connection failed" in captured.out

    def test_print_summary_error_count_in_breakdown(self, capsys):
        """Test that error count appears in the breakdown."""
        results = ActionEvalResults(
            total=3,
            passed=1,
            action_expected_passed=1,
            correct_silence=0,
            error_count=2,
        )
        results.cases = [
            ActionCaseResult(
                question="Q1",
                answer="A",
                suggested_action="Action",
                expected_action=True,
                action_passed=True,
            ),
            ActionCaseResult(
                question="Q2",
                answer="",
                suggested_action=None,
                expected_action=True,
                errors=["err1"],
            ),
            ActionCaseResult(
                question="Q3",
                answer="",
                suggested_action=None,
                expected_action=False,
                errors=["err2"],
            ),
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "Errors:                      2 failed" in captured.out

    def test_print_summary_no_error_line_when_zero(self, capsys):
        """Test that error line is hidden when no errors."""
        results = ActionEvalResults(
            total=1,
            passed=1,
            correct_silence=1,
            error_count=0,
        )
        results.cases = []

        print_summary(results)

        captured = capsys.readouterr()
        assert "Errors:" not in captured.out

    def test_print_summary_missing_action(self, capsys):
        """Test print_summary shows reason for missing action."""
        results = ActionEvalResults(
            total=1,
            passed=0,
            action_missing=1,
        )
        results.cases = [
            ActionCaseResult(
                question="Notes about Anna Lopez",
                answer="Answer",
                suggested_action=None,
                expected_action=True,
                action_passed=False,
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "Action: (none produced - expected)" in captured.out

    def test_print_summary_spurious_action(self, capsys):
        """Test print_summary shows spurious action."""
        results = ActionEvalResults(
            total=1,
            passed=0,
            spurious_action=1,
        )
        results.cases = [
            ActionCaseResult(
                question="What is Acme's plan?",
                answer="Answer",
                suggested_action="Schedule a call with the account manager",
                expected_action=False,
                action_passed=False,
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "Action: Schedule a call with the account manager" in captured.out
        assert "Answer: Answer" in captured.out

    def test_print_summary_action_metrics_shown(self, capsys):
        """Test that action metrics are shown when judged cases exist."""
        results = ActionEvalResults(
            total=2,
            passed=1,
            action_expected_passed=1,
            action_expected_failed=0,
        )
        results.avg_relevance = 0.85
        results.avg_actionability = 0.90
        results.avg_appropriateness = 0.88
        results.cases = []

        print_summary(results)

        captured = capsys.readouterr()
        assert "  Action Metrics: rel=0.85 act=0.90 app=0.88" in captured.out

    def test_print_summary_no_action_metrics_when_no_judged(self, capsys):
        """Test that action metrics are hidden when no judged cases."""
        results = ActionEvalResults(
            total=2,
            passed=2,
            action_expected_passed=0,
            action_expected_failed=0,
            correct_silence=2,
        )
        results.cases = []

        print_summary(results)

        captured = capsys.readouterr()
        assert "Action Metrics" not in captured.out


class TestMain:
    """Tests for main CLI function."""

    @patch("backend.eval.answer.action.runner.print_summary")
    @patch("backend.eval.answer.action.runner.run_action_eval")
    def test_main_calls_run_and_print(
        self,
        mock_run: MagicMock,
        mock_print: MagicMock,
    ):
        """Test main function calls run_action_eval and print_summary."""
        from backend.eval.answer.action.runner import main

        mock_run.return_value = ActionEvalResults(total=5, passed=4)

        main(limit=10)

        mock_run.assert_called_once_with(limit=10)
        mock_print.assert_called_once()
