"""Tests for backend.eval.answer.text.runner module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.eval.answer.shared.models import Question
from backend.eval.answer.text.models import TextCaseResult, TextEvalResults
from backend.eval.answer.text.runner import print_summary, run_text_eval


class TestRunTextEval:
    """Tests for run_text_eval function."""

    @patch("backend.eval.answer.text.runner.get_connection")
    @patch("backend.eval.answer.text.runner.load_questions")
    @patch("backend.eval.answer.text.runner.generate_answer")
    @patch("backend.eval.answer.text.runner.evaluate_single")
    def test_run_text_eval_basic(
        self,
        mock_evaluate: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test basic text evaluation run."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ("Answer 1", None, [{"col": 1}], None)
        mock_evaluate.return_value = {
            "answer_correctness": 0.55,
            "answer_relevancy": 0.90,
        }

        results = run_text_eval()

        assert results.total == 1
        assert results.passed == 1
        assert len(results.cases) == 1
        assert results.cases[0].answer_correctness_score == 0.55
        assert results.cases[0].answer_relevancy_score == 0.90

    @patch("backend.eval.answer.text.runner.get_connection")
    @patch("backend.eval.answer.text.runner.load_questions")
    @patch("backend.eval.answer.text.runner.generate_answer")
    def test_run_text_eval_with_error(
        self,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test text evaluation with error."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ("", None, [], "SQL error: timeout")

        results = run_text_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].errors == ["SQL error: timeout"]

    @patch("backend.eval.answer.text.runner.get_connection")
    @patch("backend.eval.answer.text.runner.load_questions")
    @patch("backend.eval.answer.text.runner.generate_answer")
    def test_run_text_eval_no_sql_results(
        self,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test text evaluation when SQL returns no results."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ("Some answer", None, None, None)

        results = run_text_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].errors == ["No SQL results - skipping RAGAS"]

    @patch("backend.eval.answer.text.runner.get_connection")
    @patch("backend.eval.answer.text.runner.load_questions")
    @patch("backend.eval.answer.text.runner.generate_answer")
    @patch("backend.eval.answer.text.runner.evaluate_single")
    def test_run_text_eval_ragas_exception(
        self,
        mock_evaluate: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test text evaluation when RAGAS raises exception."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_generate.return_value = ("Answer", None, [{"col": 1}], None)
        mock_evaluate.side_effect = ValueError("RAGAS internal error")

        results = run_text_eval()

        assert results.total == 1
        assert results.passed == 0
        assert "RAGAS failed: RAGAS internal error" in results.cases[0].errors[0]

    @patch("backend.eval.answer.text.runner.get_connection")
    @patch("backend.eval.answer.text.runner.load_questions")
    @patch("backend.eval.answer.text.runner.generate_answer")
    @patch("backend.eval.answer.text.runner.evaluate_single")
    def test_run_text_eval_with_limit(
        self,
        mock_evaluate: MagicMock,
        mock_generate: MagicMock,
        mock_load: MagicMock,
        mock_conn: MagicMock,
    ):
        """Test text evaluation with limit parameter."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
            Question(text="Q2", expected_sql="SELECT 2"),
            Question(text="Q3", expected_sql="SELECT 3"),
        ]
        mock_generate.return_value = ("Answer", None, [{}], None)
        mock_evaluate.return_value = {
            "answer_correctness": 0.55,
            "answer_relevancy": 0.90,
        }

        results = run_text_eval(limit=2)

        assert results.total == 2
        assert mock_generate.call_count == 2

    @patch("backend.eval.answer.text.runner.get_connection")
    @patch("backend.eval.answer.text.runner.load_questions")
    @patch("backend.eval.answer.text.runner.generate_answer")
    @patch("backend.eval.answer.text.runner.evaluate_single")
    def test_run_text_eval_computes_aggregates(
        self,
        mock_evaluate: MagicMock,
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
            ("Answer 1", None, [{}], None),
            ("Answer 2", None, [{}], None),
        ]
        mock_evaluate.side_effect = [
            {"answer_correctness": 0.52, "answer_relevancy": 0.90},
            {"answer_correctness": 0.58, "answer_relevancy": 0.92},
        ]

        results = run_text_eval()

        assert results.avg_answer_correctness == 0.55
        assert results.avg_answer_relevancy == 0.91


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary_passing(self, capsys):
        """Test print_summary with passing results."""
        results = TextEvalResults(total=10, passed=9)
        results.avg_answer_correctness = 0.55
        results.avg_answer_relevancy = 0.90

        print_summary(results)

        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "90.0%" in captured.out

    def test_print_summary_failing(self, capsys):
        """Test print_summary with failing results."""
        results = TextEvalResults(total=10, passed=5)
        results.cases = [
            TextCaseResult(
                question="Failed question",
                answer="Bad answer",
                answer_correctness_score=0.45,  # Below 0.50 threshold
                answer_relevancy_score=0.90,
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "Failed Cases" in captured.out

    def test_print_summary_with_error_case(self, capsys):
        """Test print_summary with a failed case that has errors."""
        results = TextEvalResults(total=10, passed=5)
        results.cases = [
            TextCaseResult(
                question="Question with error",
                answer="",
                errors=["SQL timeout", "Connection failed"],
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "Error: SQL timeout; Connection failed" in captured.out


class TestMain:
    """Tests for main CLI function."""

    @patch("backend.eval.answer.text.runner.print_summary")
    @patch("backend.eval.answer.text.runner.run_text_eval")
    def test_main_calls_run_and_print(
        self,
        mock_run: MagicMock,
        mock_print: MagicMock,
    ):
        """Test main function calls run_text_eval and print_summary."""
        from backend.eval.answer.text.runner import main

        mock_run.return_value = TextEvalResults(total=5, passed=4)

        main(limit=10)

        mock_run.assert_called_once_with(limit=10)
        mock_print.assert_called_once()
