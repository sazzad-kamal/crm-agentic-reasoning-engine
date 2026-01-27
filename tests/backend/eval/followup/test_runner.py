"""Tests for backend.eval.followup.runner module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.eval.answer.shared.models import Question
from backend.eval.followup.models import FollowupCaseResult, FollowupEvalResults
from backend.eval.followup.runner import print_summary, run_followup_eval


class TestRunFollowupEval:
    """Tests for run_followup_eval function."""

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_followup_passing(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
    ):
        """Test passing followup evaluation."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ["Follow-up 1?", "Follow-up 2?", "Follow-up 3?"]
        mock_judge.return_value = (True, 0.8, 0.7, "Good suggestions")

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 1
        assert results.cases[0].passed is True
        assert results.cases[0].relevance == 0.8
        assert results.cases[0].diversity == 0.7
        assert results.cases[0].explanation == "Good suggestions"
        assert len(results.cases[0].suggestions) == 3
        mock_judge.assert_called_once()

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_followup_failing(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
    ):
        """Test failing followup evaluation."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ["Follow-up 1?", "Follow-up 2?", "Follow-up 3?"]
        mock_judge.return_value = (False, 0.4, 0.3, "Poor suggestions")

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].passed is False
        mock_judge.assert_called_once()

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    def test_generation_error(
        self,
        mock_generate: MagicMock,
        mock_load: MagicMock,
    ):
        """Test followup evaluation with generation error."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.side_effect = ValueError("Generation failed")

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 0
        assert "Generation error" in results.cases[0].errors[0]

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_judge_exception(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
    ):
        """Test followup evaluation when judge raises exception."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ["Follow-up 1?", "Follow-up 2?", "Follow-up 3?"]
        mock_judge.side_effect = ValueError("Judge internal error")

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 0
        assert "Judge failed: Judge internal error" in results.cases[0].errors[0]

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_empty_suggestions_skips_judge(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
    ):
        """Test that empty suggestions skip the judge."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = []

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].suggestions == []
        mock_judge.assert_not_called()

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_limit_parameter(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
    ):
        """Test followup evaluation with limit parameter."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
            Question(text="Q2", expected_sql="SELECT 2"),
            Question(text="Q3", expected_sql="SELECT 3"),
        ]
        mock_generate.return_value = ["A?", "B?", "C?"]
        mock_judge.return_value = (True, 0.8, 0.7, "Good")

        results = run_followup_eval(limit=2)

        assert results.total == 2
        assert mock_generate.call_count == 2

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_computes_aggregates(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
    ):
        """Test that aggregates are computed."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
            Question(text="Q2", expected_sql="SELECT 2"),
        ]
        mock_generate.return_value = ["A?", "B?", "C?"]
        mock_judge.side_effect = [
            (True, 0.8, 0.7, "Good"),
            (True, 0.6, 0.5, "OK"),
        ]

        results = run_followup_eval()

        assert results.passed == 2
        assert results.avg_relevance == 0.7  # (0.8 + 0.6) / 2
        assert results.avg_diversity == 0.6  # (0.7 + 0.5) / 2

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_use_hardcoded_tree_parameter(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
    ):
        """Test use_hardcoded_tree parameter is passed through."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ["A?", "B?", "C?"]
        mock_judge.return_value = (True, 0.8, 0.7, "Good")

        run_followup_eval(use_hardcoded_tree=False)

        mock_generate.assert_called_once_with(
            question="Q1",
            use_hardcoded_tree=False,
        )


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary_passing(self, capsys):
        """Test print_summary with passing results."""
        results = FollowupEvalResults(total=10, passed=9)
        results.avg_relevance = 0.85
        results.avg_diversity = 0.70
        results.cases = []

        print_summary(results)

        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "90.0%" in captured.out
        assert ">=80.0% SLO" in captured.out
        assert "rel=0.85 div=0.70" in captured.out

    def test_print_summary_failing(self, capsys):
        """Test print_summary with failing results."""
        results = FollowupEvalResults(total=10, passed=5)
        results.avg_relevance = 0.50
        results.avg_diversity = 0.40
        results.cases = [
            FollowupCaseResult(
                question="Failed question",
                suggestions=["Q1?", "Q2?", "Q3?"],
                relevance=0.3,
                diversity=0.2,
                explanation="Poor quality",
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "Failed Cases" in captured.out
        assert "rel=0.30 div=0.20" in captured.out
        assert "Judge: Poor quality" in captured.out

    def test_print_summary_with_error_case(self, capsys):
        """Test print_summary with error cases."""
        results = FollowupEvalResults(total=10, passed=5)
        results.cases = [
            FollowupCaseResult(
                question="Question with error",
                errors=["Generation error: timeout"],
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "Error Cases (1)" in captured.out
        assert "Error: Generation error: timeout" in captured.out

    def test_print_summary_no_error_cases(self, capsys):
        """Test print_summary hides error section when no errors."""
        results = FollowupEvalResults(total=2, passed=2)
        results.cases = [
            FollowupCaseResult(question="Q1", passed=True),
            FollowupCaseResult(question="Q2", passed=True),
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "Error Cases" not in captured.out

    def test_print_summary_no_failed_cases(self, capsys):
        """Test print_summary hides failed section when all pass."""
        results = FollowupEvalResults(total=2, passed=2)
        results.cases = [
            FollowupCaseResult(question="Q1", passed=True),
            FollowupCaseResult(question="Q2", passed=True),
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "Failed Cases" not in captured.out

    def test_print_summary_failed_shows_suggestions(self, capsys):
        """Test print_summary shows suggestions for failed cases."""
        results = FollowupEvalResults(total=1, passed=0)
        results.cases = [
            FollowupCaseResult(
                question="What deals does Acme have?",
                suggestions=["Follow-up 1?", "Follow-up 2?"],
                relevance=0.3,
                diversity=0.2,
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "- Follow-up 1?" in captured.out
        assert "- Follow-up 2?" in captured.out


class TestMain:
    """Tests for main CLI function."""

    @patch("backend.eval.followup.runner.print_summary")
    @patch("backend.eval.followup.runner.run_followup_eval")
    def test_main_calls_run_and_print(
        self,
        mock_run: MagicMock,
        mock_print: MagicMock,
    ):
        """Test main function calls run_followup_eval and print_summary."""
        from backend.eval.followup.runner import main

        mock_run.return_value = FollowupEvalResults(total=5, passed=4)

        main(limit=10, no_tree=False)

        mock_run.assert_called_once_with(limit=10, use_hardcoded_tree=True)
        mock_print.assert_called_once()

    @patch("backend.eval.followup.runner.print_summary")
    @patch("backend.eval.followup.runner.run_followup_eval")
    def test_main_no_tree_flag(
        self,
        mock_run: MagicMock,
        mock_print: MagicMock,
    ):
        """Test main passes no_tree flag correctly."""
        from backend.eval.followup.runner import main

        mock_run.return_value = FollowupEvalResults(total=5, passed=4)

        main(limit=None, no_tree=True)

        mock_run.assert_called_once_with(limit=None, use_hardcoded_tree=False)
