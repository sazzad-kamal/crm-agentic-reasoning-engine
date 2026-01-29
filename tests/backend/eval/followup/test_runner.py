"""Tests for backend.eval.followup.runner module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.eval.answer.shared.models import Question
from backend.eval.followup.models import FollowupCaseResult, FollowupEvalResults
from backend.eval.followup.runner import print_summary, run_followup_eval


class TestRunFollowupEval:
    """Tests for run_followup_eval function."""

    @patch("backend.eval.followup.runner._check_answerability")
    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_followup_passing(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
        mock_answerability: MagicMock,
    ):
        """Test passing followup evaluation."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_answer.return_value = ("Acme has 3 deals.", [], None)
        mock_generate.return_value = ["Follow-up 1?", "Follow-up 2?", "Follow-up 3?"]
        mock_judge.return_value = (True, 0.8, 0.6, 0.7, "Good suggestions")
        mock_answerability.return_value = (3, 1.0)

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 1
        assert results.cases[0].passed is True
        assert results.cases[0].answer == "Acme has 3 deals."
        assert results.cases[0].question_relevance == 0.8
        assert results.cases[0].answer_grounding == 0.6
        assert results.cases[0].diversity == 0.7
        assert results.cases[0].explanation == "Good suggestions"
        assert len(results.cases[0].suggestions) == 3
        assert results.cases[0].answerable_count == 3
        assert results.cases[0].answerability == 1.0
        mock_judge.assert_called_once()

    @patch("backend.eval.followup.runner._check_answerability")
    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_followup_failing(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
        mock_answerability: MagicMock,
    ):
        """Test failing followup evaluation."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_answer.return_value = ("Answer text.", [], None)
        mock_generate.return_value = ["Follow-up 1?", "Follow-up 2?", "Follow-up 3?"]
        mock_judge.return_value = (False, 0.4, 0.2, 0.3, "Poor suggestions")
        mock_answerability.return_value = (1, 0.33)

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].passed is False
        mock_judge.assert_called_once()

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    def test_generation_error(
        self,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
    ):
        """Test followup evaluation with generation error."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_answer.return_value = ("Answer text.", [], None)
        mock_generate.side_effect = ValueError("Generation failed")

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 0
        assert "Generation error" in results.cases[0].errors[0]

    @patch("backend.eval.followup.runner._check_answerability")
    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_judge_exception(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
        mock_answerability: MagicMock,
    ):
        """Test followup evaluation when judge raises exception."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_answer.return_value = ("Answer text.", [], None)
        mock_generate.return_value = ["Follow-up 1?", "Follow-up 2?", "Follow-up 3?"]
        mock_judge.side_effect = ValueError("Judge internal error")
        mock_answerability.return_value = (2, 0.67)

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 0
        assert "Judge failed: Judge internal error" in results.cases[0].errors[0]

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_empty_suggestions_skips_judge(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
    ):
        """Test that empty suggestions skip the judge."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_answer.return_value = ("Answer text.", [], None)
        mock_generate.return_value = []

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].suggestions == []
        mock_judge.assert_not_called()

    @patch("backend.eval.followup.runner._check_answerability")
    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_limit_parameter(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
        mock_answerability: MagicMock,
    ):
        """Test followup evaluation with limit parameter."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
            Question(text="Q2", expected_sql="SELECT 2"),
            Question(text="Q3", expected_sql="SELECT 3"),
        ]
        mock_answer.return_value = ("Answer.", [], None)
        mock_generate.return_value = ["A?", "B?", "C?"]
        mock_judge.return_value = (True, 0.8, 0.6, 0.7, "Good")
        mock_answerability.return_value = (3, 1.0)

        results = run_followup_eval(limit=2)

        assert results.total == 2
        assert mock_generate.call_count == 2

    @patch("backend.eval.followup.runner._check_answerability")
    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_computes_aggregates(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
        mock_answerability: MagicMock,
    ):
        """Test that aggregates are computed."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
            Question(text="Q2", expected_sql="SELECT 2"),
        ]
        mock_answer.return_value = ("Answer.", [], None)
        mock_generate.return_value = ["A?", "B?", "C?"]
        mock_judge.side_effect = [
            (True, 0.8, 0.6, 0.7, "Good"),
            (True, 0.6, 0.4, 0.5, "OK"),
        ]
        mock_answerability.side_effect = [(3, 1.0), (2, 0.67)]

        results = run_followup_eval()

        assert results.passed == 2
        assert results.avg_question_relevance == 0.7
        assert results.avg_answer_grounding == 0.5
        assert results.avg_diversity == 0.6

    @patch("backend.eval.followup.runner._check_answerability")
    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_always_uses_llm_path(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
        mock_answerability: MagicMock,
    ):
        """Test eval always uses LLM path (no hardcoded tree)."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_answer.return_value = ("Acme answer.", [], None)
        mock_generate.return_value = ["A?", "B?", "C?"]
        mock_judge.return_value = (True, 0.8, 0.6, 0.7, "Good")
        mock_answerability.return_value = (3, 1.0)

        run_followup_eval()

        mock_generate.assert_called_once_with(
            question="Q1",
            answer="Acme answer.",
            use_hardcoded_tree=False,
        )

    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    def test_answer_error_skips_generation(
        self,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
    ):
        """Test that answer generation error skips follow-up generation."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_answer.return_value = ("", [], "SQL error: table not found")

        results = run_followup_eval()

        assert results.total == 1
        assert results.passed == 0
        assert "Answer error" in results.cases[0].errors[0]
        mock_generate.assert_not_called()

    @patch("backend.eval.followup.runner._check_answerability")
    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_answer_passed_to_judge(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
        mock_answerability: MagicMock,
    ):
        """Test that answer is passed to the judge."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_answer.return_value = ("Acme has 3 deals.", [], None)
        mock_generate.return_value = ["A?", "B?", "C?"]
        mock_judge.return_value = (True, 0.8, 0.6, 0.7, "Good")
        mock_answerability.return_value = (3, 1.0)

        run_followup_eval()

        mock_judge.assert_called_once_with(
            "Q1", ["A?", "B?", "C?"], answer="Acme has 3 deals."
        )

    @patch("backend.eval.followup.runner._check_answerability")
    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_answerability_populated(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
        mock_answerability: MagicMock,
    ):
        """Test answerability fields are populated on case results."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
        ]
        mock_answer.return_value = ("Answer.", [], None)
        mock_generate.return_value = ["A?", "B?", "C?"]
        mock_judge.return_value = (True, 0.8, 0.6, 0.7, "Good")
        mock_answerability.return_value = (2, 0.67)

        results = run_followup_eval()

        assert results.cases[0].answerable_count == 2
        assert results.cases[0].answerability == 0.67

    @patch("backend.eval.followup.runner._check_answerability")
    @patch("backend.eval.followup.runner.load_questions")
    @patch("backend.eval.followup.runner.get_connection")
    @patch("backend.eval.followup.runner.generate_answer")
    @patch("backend.eval.followup.runner.generate_follow_up_suggestions")
    @patch("backend.eval.followup.runner.judge_followup_suggestions")
    def test_answerability_aggregated(
        self,
        mock_judge: MagicMock,
        mock_generate: MagicMock,
        mock_answer: MagicMock,
        mock_conn: MagicMock,
        mock_load: MagicMock,
        mock_answerability: MagicMock,
    ):
        """Test avg_answerability is computed in aggregates."""
        mock_load.return_value = [
            Question(text="Q1", expected_sql="SELECT 1"),
            Question(text="Q2", expected_sql="SELECT 2"),
        ]
        mock_answer.return_value = ("Answer.", [], None)
        mock_generate.return_value = ["A?", "B?", "C?"]
        mock_judge.side_effect = [
            (True, 0.8, 0.6, 0.7, "Good"),
            (True, 0.6, 0.4, 0.5, "OK"),
        ]
        mock_answerability.side_effect = [(3, 1.0), (1, 0.33)]

        results = run_followup_eval()

        assert 0.6 < results.avg_answerability < 0.7


class TestCheckAnswerability:
    """Tests for _check_answerability function."""

    @patch("backend.eval.followup.runner.execute_sql")
    @patch("backend.eval.followup.runner.get_sql_plan")
    def test_all_answerable(self, mock_plan: MagicMock, mock_execute: MagicMock):
        """All suggestions return data."""
        from backend.eval.followup.runner import _check_answerability

        mock_plan.return_value = MagicMock(sql="SELECT 1")
        mock_execute.return_value = ([{"col": "val"}], None)

        count, ratio = _check_answerability(["Q1?", "Q2?", "Q3?"], MagicMock())

        assert count == 3
        assert ratio == 1.0

    @patch("backend.eval.followup.runner.execute_sql")
    @patch("backend.eval.followup.runner.get_sql_plan")
    def test_partial_answerable(self, mock_plan: MagicMock, mock_execute: MagicMock):
        """Some suggestions return empty results."""
        from backend.eval.followup.runner import _check_answerability

        mock_plan.return_value = MagicMock(sql="SELECT 1")
        mock_execute.side_effect = [
            ([{"col": "val"}], None),
            ([], None),
            ([{"col": "val"}], None),
        ]

        count, ratio = _check_answerability(["Q1?", "Q2?", "Q3?"], MagicMock())

        assert count == 2
        assert ratio == 2 / 3

    @patch("backend.eval.followup.runner.execute_sql")
    @patch("backend.eval.followup.runner.get_sql_plan")
    def test_sql_error_not_answerable(self, mock_plan: MagicMock, mock_execute: MagicMock):
        """SQL errors count as not answerable."""
        from backend.eval.followup.runner import _check_answerability

        mock_plan.return_value = MagicMock(sql="SELECT 1")
        mock_execute.return_value = ([], "SQL error")

        count, ratio = _check_answerability(["Q1?"], MagicMock())

        assert count == 0
        assert ratio == 0.0

    @patch("backend.eval.followup.runner.get_sql_plan")
    def test_planner_exception_not_answerable(self, mock_plan: MagicMock):
        """Planner exceptions count as not answerable."""
        from backend.eval.followup.runner import _check_answerability

        mock_plan.side_effect = ValueError("Planner error")

        count, ratio = _check_answerability(["Q1?"], MagicMock())

        assert count == 0
        assert ratio == 0.0

    def test_empty_suggestions(self):
        """Empty suggestions returns zero."""
        from backend.eval.followup.runner import _check_answerability

        count, ratio = _check_answerability([], MagicMock())

        assert count == 0
        assert ratio == 0.0


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary_passing(self, capsys):
        """Test print_summary with passing results."""
        results = FollowupEvalResults(total=10, passed=9)
        results.avg_question_relevance = 0.85
        results.avg_answer_grounding = 0.60
        results.avg_diversity = 0.70
        results.avg_answerability = 0.90
        results.cases = []

        print_summary(results)

        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "90.0%" in captured.out
        assert ">=80.0% SLO" in captured.out
        assert "qrel=0.85" in captured.out
        assert "agrnd=0.60" in captured.out
        assert "div=0.70" in captured.out
        assert "Answerability: 90.0%" in captured.out

    def test_print_summary_failing(self, capsys):
        """Test print_summary with failing results."""
        results = FollowupEvalResults(total=10, passed=5)
        results.avg_question_relevance = 0.50
        results.avg_answer_grounding = 0.30
        results.avg_diversity = 0.40
        results.avg_answerability = 0.50
        results.cases = [
            FollowupCaseResult(
                question="Failed question",
                suggestions=["Q1?", "Q2?", "Q3?"],
                question_relevance=0.3,
                answer_grounding=0.1,
                diversity=0.2,
                explanation="Poor quality",
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "FAIL" in captured.out
        assert "Failed Cases" in captured.out
        assert "Answerability: 50.0%" in captured.out

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
                question_relevance=0.3,
                answer_grounding=0.1,
                diversity=0.2,
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "- Follow-up 1?" in captured.out
        assert "- Follow-up 2?" in captured.out

    def test_print_summary_failed_shows_answerability(self, capsys):
        """Test print_summary shows answerability for failed cases."""
        results = FollowupEvalResults(total=1, passed=0)
        results.cases = [
            FollowupCaseResult(
                question="Test?",
                suggestions=["A?", "B?", "C?"],
                question_relevance=0.3,
                answer_grounding=0.1,
                diversity=0.2,
                answerable_count=1,
            )
        ]

        print_summary(results)

        captured = capsys.readouterr()
        assert "Answerability: 1/3" in captured.out


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

        main(limit=10)

        mock_run.assert_called_once_with(limit=10)
        mock_print.assert_called_once()
