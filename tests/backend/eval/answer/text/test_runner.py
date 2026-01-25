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
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
            "answer_correctness": 0.85,
        }

        results = run_text_eval()

        assert results.total == 1
        assert results.passed == 1
        assert len(results.cases) == 1
        assert results.cases[0].faithfulness_score == 0.8
        assert results.cases[0].relevance_score == 0.9

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
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
            "answer_correctness": 0.85,
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
            {"faithfulness": 0.7, "answer_relevancy": 0.8, "answer_correctness": 0.75},
            {"faithfulness": 0.9, "answer_relevancy": 0.85, "answer_correctness": 0.95},
        ]

        results = run_text_eval()

        assert results.avg_faithfulness == 0.8
        assert results.avg_relevance == 0.825


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary_passing(self, capsys):
        """Test print_summary with passing results."""
        results = TextEvalResults(total=10, passed=9)
        results.avg_faithfulness = 0.85
        results.avg_relevance = 0.90
        results.avg_answer_correctness = 0.88

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
                faithfulness_score=0.3,
                relevance_score=0.4,
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "Failed Cases" in captured.out
