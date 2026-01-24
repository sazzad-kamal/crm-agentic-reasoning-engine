"""Tests for backend.eval.answer module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.eval.answer.models import CaseResult, EvalResults, Question


# =============================================================================
# Tests for models.py
# =============================================================================


class TestQuestion:
    """Tests for Question dataclass."""

    def test_question_basic(self):
        """Test basic Question creation."""
        q = Question(text="What is the status?", difficulty=1, expected_sql="SELECT status FROM companies")
        assert q.text == "What is the status?"
        assert q.difficulty == 1
        assert q.expected_sql == "SELECT status FROM companies"

    def test_question_with_expected_answer(self):
        """Test Question with expected_answer field."""
        q = Question(
            text="What is the plan?",
            difficulty=1,
            expected_sql="SELECT plan FROM companies",
            expected_answer="The plan is Enterprise.",
        )
        assert q.expected_answer == "The plan is Enterprise."

    def test_question_expected_answer_optional(self):
        """Test expected_answer is optional."""
        q = Question(text="Test", difficulty=1, expected_sql="SELECT 1")
        assert q.expected_answer == ""


class TestCaseResult:
    """Tests for CaseResult dataclass."""

    def test_case_result_basic(self):
        """Test basic CaseResult creation."""
        case = CaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action=None,
            latency_ms=100,
        )
        assert case.question == "Test question"
        assert case.answer == "Test answer"
        assert case.suggested_action is None
        assert case.latency_ms == 100
        assert case.errors == []

    def test_case_result_with_action(self):
        """Test CaseResult with suggested action."""
        case = CaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action="Schedule a follow-up call",
            latency_ms=150,
            action_relevance=0.8,
            action_actionability=0.9,
            action_appropriateness=0.85,
            action_passed=True,
        )
        assert case.suggested_action == "Schedule a follow-up call"
        assert case.action_passed is True

    def test_case_result_with_ragas_scores(self):
        """Test CaseResult with RAGAS scores."""
        case = CaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action=None,
            latency_ms=100,
            faithfulness_score=0.8,
            relevance_score=0.9,
            answer_correctness_score=0.85,
        )
        assert case.faithfulness_score == 0.8
        assert case.relevance_score == 0.9
        assert case.answer_correctness_score == 0.85

    def test_case_result_passed_computation_success(self):
        """Test passed property when RAGAS scores are good."""
        case = CaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action=None,
            latency_ms=100,
            faithfulness_score=0.8,
            relevance_score=0.9,
        )
        assert case.passed is True

    def test_case_result_passed_computation_failure_faithfulness(self):
        """Test passed property when faithfulness is low."""
        case = CaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action=None,
            latency_ms=100,
            faithfulness_score=0.5,  # Below 0.6 threshold
            relevance_score=0.9,
        )
        assert case.passed is False

    def test_case_result_passed_computation_failure_relevance(self):
        """Test passed property when relevance is low."""
        case = CaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action=None,
            latency_ms=100,
            faithfulness_score=0.8,
            relevance_score=0.5,  # Below 0.6 threshold
        )
        assert case.passed is False

    def test_case_result_passed_computation_with_action(self):
        """Test passed property considers action when present."""
        case = CaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action="Call customer",
            latency_ms=100,
            faithfulness_score=0.8,
            relevance_score=0.9,
            action_passed=False,  # Action failed
        )
        assert case.passed is False

    def test_case_result_passed_computation_with_errors(self):
        """Test passed property is False when errors present."""
        case = CaseResult(
            question="Test question",
            answer="Test answer",
            suggested_action=None,
            latency_ms=100,
            faithfulness_score=0.8,
            relevance_score=0.9,
            errors=["SQL error"],
        )
        assert case.passed is False


class TestEvalResults:
    """Tests for EvalResults dataclass."""

    def test_eval_results_defaults(self):
        """Test EvalResults default values."""
        results = EvalResults()
        assert results.total == 0
        assert results.passed == 0
        assert results.cases == []

    def test_failed_property(self):
        """Test failed property calculation."""
        results = EvalResults(total=10, passed=7)
        assert results.failed == 3

    def test_pass_rate_property(self):
        """Test pass_rate property calculation."""
        results = EvalResults(total=10, passed=8)
        assert results.pass_rate == 0.8

    def test_pass_rate_zero_total(self):
        """Test pass_rate with zero total."""
        results = EvalResults(total=0, passed=0)
        assert results.pass_rate == 0.0

    def test_compute_aggregates_empty_cases(self):
        """Test compute_aggregates with empty cases."""
        results = EvalResults()
        results.compute_aggregates()
        # Should not raise, values stay at defaults
        assert results.avg_latency_ms == 0.0

    def test_compute_aggregates_basic(self):
        """Test compute_aggregates computes averages."""
        results = EvalResults()
        results.cases = [
            CaseResult(
                question="Q1",
                answer="A1",
                suggested_action=None,
                latency_ms=100,
                faithfulness_score=0.7,
                relevance_score=0.8,
                answer_correctness_score=0.75,
            ),
            CaseResult(
                question="Q2",
                answer="A2",
                suggested_action=None,
                latency_ms=200,
                faithfulness_score=0.9,
                relevance_score=0.85,
                answer_correctness_score=0.95,
            ),
        ]
        results.compute_aggregates()

        assert results.avg_latency_ms == 150.0
        assert results.avg_faithfulness == 0.8
        assert results.avg_relevance == 0.825
        assert results.avg_answer_correctness == 0.85

    def test_compute_aggregates_with_actions(self):
        """Test compute_aggregates computes action metrics."""
        results = EvalResults()
        results.cases = [
            CaseResult(
                question="Q1",
                answer="A1",
                suggested_action="Call customer",
                latency_ms=100,
                action_relevance=0.8,
                action_actionability=0.9,
                action_appropriateness=0.85,
                action_passed=True,
            ),
            CaseResult(
                question="Q2",
                answer="A2",
                suggested_action="Send email",
                latency_ms=150,
                action_relevance=0.7,
                action_actionability=0.6,
                action_appropriateness=0.65,
                action_passed=False,
            ),
            CaseResult(
                question="Q3",
                answer="A3",
                suggested_action=None,  # No action
                latency_ms=80,
            ),
        ]
        results.compute_aggregates()

        # Only action cases counted for action metrics
        assert results.avg_action_relevance == 0.75
        assert results.avg_action_actionability == 0.75
        assert results.avg_action_appropriateness == 0.75
        assert results.action_pass_rate == 0.5


# =============================================================================
# Tests for judge.py
# =============================================================================


class TestJudgeSuggestedAction:
    """Tests for judge_suggested_action function."""

    def _mock_chain(self, result):
        """Create a mock chain that returns the given ActionJudgeResult."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = result
        return mock_chain

    def test_judge_suggested_action_passed(self, monkeypatch):
        """Test successful judgment that passes."""
        import backend.eval.answer.judge as judge_module
        from backend.eval.answer.judge import ActionJudgeResult, judge_suggested_action

        mock_chain = self._mock_chain(
            ActionJudgeResult(
                relevance=0.9,
                actionability=0.85,
                appropriateness=0.8,
                passed=True,
                explanation="Good action",
            )
        )
        monkeypatch.setattr(judge_module, "create_openai_chain", lambda **kwargs: mock_chain)

        passed, rel, act, app, exp = judge_suggested_action(
            question="What is the status?",
            answer="The status is active.",
            action="Schedule a review call",
        )

        assert passed is True
        assert rel == 0.9
        assert act == 0.85
        assert app == 0.8
        assert exp == "Good action"

    def test_judge_suggested_action_failed(self, monkeypatch):
        """Test judgment that fails."""
        import backend.eval.answer.judge as judge_module
        from backend.eval.answer.judge import ActionJudgeResult, judge_suggested_action

        mock_chain = self._mock_chain(
            ActionJudgeResult(
                relevance=0.5,
                actionability=0.4,
                appropriateness=0.3,
                passed=False,
                explanation="Vague action",
            )
        )
        monkeypatch.setattr(judge_module, "create_openai_chain", lambda **kwargs: mock_chain)

        passed, rel, act, app, exp = judge_suggested_action(
            question="What is the status?",
            answer="The status is active.",
            action="Follow up",
        )

        assert passed is False
        assert rel == 0.5
        assert "Vague" in exp

    def test_judge_suggested_action_api_error(self, monkeypatch):
        """Test API error returns failure."""
        import backend.eval.answer.judge as judge_module
        from backend.eval.answer.judge import judge_suggested_action

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API connection error")
        monkeypatch.setattr(judge_module, "create_openai_chain", lambda **kwargs: mock_chain)

        passed, rel, act, app, exp = judge_suggested_action(
            question="Test",
            answer="Test answer",
            action="Test action",
        )

        assert passed is False
        assert rel == 0.0
        assert "Judge error" in exp


# =============================================================================
# Tests for runner.py
# =============================================================================


class TestLoadQuestions:
    """Tests for load_questions function."""

    def test_load_questions_all(self, monkeypatch, tmp_path):
        """Test loading all questions."""
        yaml_content = """
questions:
  - text: "Question 1"
    difficulty: 1
    expected_sql: "SELECT 1"
    expected_answer: "Answer 1"
  - text: "Question 2"
    difficulty: 2
    expected_sql: "SELECT 2"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)

        import backend.eval.answer.runner as runner_module

        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        from backend.eval.answer.runner import load_questions

        questions = load_questions()
        assert len(questions) == 2
        assert questions[0].text == "Question 1"
        assert questions[0].expected_sql == "SELECT 1"
        assert questions[0].expected_answer == "Answer 1"
        assert questions[1].expected_answer == ""  # Default


class TestRunAnswerEval:
    """Tests for run_answer_eval function."""

    def test_run_answer_eval_basic(self, monkeypatch, tmp_path):
        """Test basic answer evaluation run."""
        import backend.eval.answer.runner as runner_module

        yaml_content = """
questions:
  - text: "Test question"
    difficulty: 1
    expected_sql: "SELECT 1"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        # Mock get_connection
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(1,)]
        mock_result.description = [("value",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))

        # Mock call_answer_chain
        monkeypatch.setattr(
            runner_module, "call_answer_chain", MagicMock(return_value="The answer is 1.")
        )

        # Mock extract_suggested_action
        monkeypatch.setattr(
            runner_module, "extract_suggested_action", MagicMock(return_value=("The answer is 1.", None))
        )

        # Mock RAGAS evaluate_single
        monkeypatch.setattr(
            runner_module,
            "evaluate_single",
            MagicMock(return_value={
                "answer_relevancy": 0.9,
                "faithfulness": 0.8,
                "answer_correctness": 0.85,
            }),
        )

        from backend.eval.answer.runner import run_answer_eval

        results = run_answer_eval(verbose=False)

        assert results.total == 1
        assert results.passed == 1
        assert len(results.cases) == 1
        assert results.cases[0].passed is True

    def test_run_answer_eval_with_limit(self, monkeypatch, tmp_path):
        """Test answer evaluation with limit."""
        import backend.eval.answer.runner as runner_module

        yaml_content = "questions:\n" + "\n".join(
            [f'  - text: "Question {i}"\n    difficulty: 1\n    expected_sql: "SELECT {i}"' for i in range(10)]
        )
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(1,)]
        mock_result.description = [("v",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))

        monkeypatch.setattr(runner_module, "call_answer_chain", MagicMock(return_value="Answer"))
        monkeypatch.setattr(runner_module, "extract_suggested_action", MagicMock(return_value=("Answer", None)))
        monkeypatch.setattr(
            runner_module,
            "evaluate_single",
            MagicMock(return_value={"answer_relevancy": 0.9, "faithfulness": 0.8, "answer_correctness": 0.85}),
        )

        from backend.eval.answer.runner import run_answer_eval

        results = run_answer_eval(limit=3)

        assert results.total == 3
        assert len(results.cases) == 3

    def test_run_answer_eval_sql_error(self, monkeypatch, tmp_path):
        """Test handling SQL execution errors."""
        import backend.eval.answer.runner as runner_module

        yaml_content = """
questions:
  - text: "Test"
    difficulty: 1
    expected_sql: "INVALID SQL"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("SQL syntax error")
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))

        # Mock execute_sql to return error
        monkeypatch.setattr(
            runner_module, "execute_sql", MagicMock(return_value=([], "SQL syntax error"))
        )

        from backend.eval.answer.runner import run_answer_eval

        results = run_answer_eval(verbose=False)

        assert results.total == 1
        assert results.passed == 0
        assert any("SQL error" in e for e in results.cases[0].errors)

    def test_run_answer_eval_with_action(self, monkeypatch, tmp_path):
        """Test evaluation with suggested action."""
        import backend.eval.answer.runner as runner_module

        yaml_content = """
questions:
  - text: "What is the status?"
    difficulty: 1
    expected_sql: "SELECT status FROM companies"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("Active",)]
        mock_result.description = [("status",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))

        monkeypatch.setattr(
            runner_module, "call_answer_chain", MagicMock(return_value="Status is Active.\n\nSuggested action: Call customer")
        )
        monkeypatch.setattr(
            runner_module, "extract_suggested_action", MagicMock(return_value=("Status is Active.", "Call customer"))
        )
        monkeypatch.setattr(
            runner_module,
            "evaluate_single",
            MagicMock(return_value={"answer_relevancy": 0.9, "faithfulness": 0.8, "answer_correctness": 0.85}),
        )
        monkeypatch.setattr(
            runner_module, "judge_suggested_action", MagicMock(return_value=(True, 0.9, 0.8, 0.85, "Good"))
        )

        from backend.eval.answer.runner import run_answer_eval

        results = run_answer_eval(verbose=False)

        assert results.cases[0].suggested_action == "Call customer"
        assert results.cases[0].action_passed is True


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary_basic(self):
        """Test basic summary printing."""
        from backend.eval.answer.runner import print_summary

        results = EvalResults(total=10, passed=9)
        results.cases = [
            CaseResult(
                question="Q",
                answer="A",
                suggested_action=None,
                latency_ms=100,
                faithfulness_score=0.8,
                relevance_score=0.9,
            )
            for _ in range(9)
        ]
        results.cases.append(
            CaseResult(
                question="Q",
                answer="A",
                suggested_action=None,
                latency_ms=100,
                faithfulness_score=0.5,
                relevance_score=0.5,
            )
        )
        results.compute_aggregates()

        # Should not raise
        print_summary(results)

    def test_print_summary_with_failures(self):
        """Test summary printing with failed cases."""
        from backend.eval.answer.runner import print_summary

        results = EvalResults(total=3, passed=1)
        results.cases = [
            CaseResult(
                question="Passed question",
                answer="Good answer",
                suggested_action=None,
                latency_ms=100,
                faithfulness_score=0.8,
                relevance_score=0.9,
            ),
            CaseResult(
                question="Failed question 1",
                answer="Bad answer",
                suggested_action=None,
                latency_ms=100,
                faithfulness_score=0.3,
                relevance_score=0.4,
            ),
            CaseResult(
                question="Failed question 2",
                answer="",
                suggested_action=None,
                latency_ms=100,
                errors=["SQL error"],
            ),
        ]
        results.compute_aggregates()

        # Should not raise
        print_summary(results)

    def test_print_summary_many_failures(self):
        """Test summary printing with more than 10 failures shows truncation message."""
        from backend.eval.answer.runner import print_summary

        results = EvalResults(total=15, passed=0)
        results.cases = [
            CaseResult(
                question=f"Failed question {i}",
                answer=f"Answer {i}",
                suggested_action=None,
                latency_ms=100,
                faithfulness_score=0.3,
                relevance_score=0.4,
            )
            for i in range(15)
        ]
        results.compute_aggregates()

        # Should not raise, should show "... and X more failures"
        print_summary(results)

    def test_print_summary_action_failure_only(self, capsys):
        """Test summary shows action details when only action fails."""
        from backend.eval.answer.runner import print_summary

        results = EvalResults(total=1, passed=0)
        results.cases = [
            CaseResult(
                question="Schedule follow-up for pending deals",
                answer="There are 5 pending deals...",
                suggested_action="Send email to all contacts",
                latency_ms=100,
                faithfulness_score=0.85,  # RAGAS passes
                relevance_score=0.82,
                action_relevance=0.45,
                action_actionability=0.30,
                action_appropriateness=0.55,
                action_passed=False,
            ),
        ]
        results.compute_aggregates()

        print_summary(results)
        captured = capsys.readouterr()

        # Extract failed cases section
        failed_section = captured.out.split("Failed Cases")[1]

        # Should show action metrics and suggested action, but NOT RAGAS scores in failure
        assert "Action: rel=0.45" in failed_section
        assert "Suggested: Send email" in failed_section
        assert "   RAGAS:" not in failed_section  # RAGAS passed, don't show in failure

    def test_print_summary_ragas_failure_only(self, capsys):
        """Test summary shows RAGAS and answer when only RAGAS fails."""
        from backend.eval.answer.runner import print_summary

        results = EvalResults(total=1, passed=0)
        results.cases = [
            CaseResult(
                question="What is the revenue?",
                answer="Revenue is $1M...",
                suggested_action=None,  # No action
                latency_ms=100,
                faithfulness_score=0.45,  # RAGAS fails
                relevance_score=0.50,
            ),
        ]
        results.compute_aggregates()

        print_summary(results)
        captured = capsys.readouterr()

        # Extract failed cases section
        failed_section = captured.out.split("Failed Cases")[1]

        assert "RAGAS: F=0.45 R=0.50" in failed_section
        assert "Answer: Revenue is $1M" in failed_section
        assert "   Action:" not in failed_section  # No action, don't show

    def test_print_summary_both_failures(self, capsys):
        """Test summary shows all 4 when both RAGAS and action fail."""
        from backend.eval.answer.runner import print_summary

        results = EvalResults(total=1, passed=0)
        results.cases = [
            CaseResult(
                question="Schedule follow-up for deals",
                answer="There are pending deals...",
                suggested_action="Send email",
                latency_ms=100,
                faithfulness_score=0.45,  # RAGAS fails
                relevance_score=0.50,
                action_relevance=0.30,  # Action fails
                action_actionability=0.25,
                action_appropriateness=0.40,
                action_passed=False,
            ),
        ]
        results.compute_aggregates()

        print_summary(results)
        captured = capsys.readouterr()

        # Extract failed cases section
        failed_section = captured.out.split("Failed Cases")[1]

        # Should show all 4: RAGAS, answer, action metrics, suggested action
        assert "RAGAS: F=0.45 R=0.50" in failed_section
        assert "Answer: There are pending" in failed_section
        assert "Action: rel=0.30" in failed_section
        assert "Suggested: Send email" in failed_section


class TestEvalCaseExceptionHandling:
    """Tests for exception handling in _eval_case."""

    def test_eval_case_general_exception(self, monkeypatch, tmp_path):
        """Test handling general exceptions in _eval_case."""
        import backend.eval.answer.runner as runner_module

        yaml_content = """
questions:
  - text: "Test question"
    difficulty: 1
    expected_sql: "SELECT 1"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        # Mock get_connection to return a working conn
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(1,)]
        mock_result.description = [("value",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))

        # Mock call_answer_chain to raise an exception
        monkeypatch.setattr(
            runner_module, "call_answer_chain", MagicMock(side_effect=Exception("LLM API Error"))
        )

        from backend.eval.answer.runner import run_answer_eval

        results = run_answer_eval(verbose=False)

        assert results.total == 1
        assert results.passed == 0
        assert any("Error: LLM API Error" in e for e in results.cases[0].errors)


class TestVerboseMode:
    """Tests for verbose mode output."""

    def test_run_answer_eval_verbose_pass(self, monkeypatch, tmp_path, capsys):
        """Test verbose output for passing case."""
        import backend.eval.answer.runner as runner_module

        yaml_content = """
questions:
  - text: "Test question"
    difficulty: 1
    expected_sql: "SELECT 1"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(1,)]
        mock_result.description = [("value",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))

        monkeypatch.setattr(runner_module, "call_answer_chain", MagicMock(return_value="The answer is 1."))
        monkeypatch.setattr(runner_module, "extract_suggested_action", MagicMock(return_value=("The answer is 1.", None)))
        monkeypatch.setattr(
            runner_module,
            "evaluate_single",
            MagicMock(return_value={"answer_relevancy": 0.9, "faithfulness": 0.8, "answer_correctness": 0.85}),
        )

        from backend.eval.answer.runner import run_answer_eval

        results = run_answer_eval(verbose=True)

        assert results.total == 1
        assert results.passed == 1


# =============================================================================
# Tests for main CLI
# =============================================================================


class TestMainCli:
    """Tests for CLI main function."""

    def test_main_basic(self, monkeypatch):
        """Test main function runs without error."""
        import backend.eval.answer.runner as runner_module

        mock_results = EvalResults(total=1, passed=1)

        monkeypatch.setattr(runner_module, "run_answer_eval", MagicMock(return_value=mock_results))
        monkeypatch.setattr(runner_module, "print_summary", MagicMock())

        # Should not raise
        runner_module.main(limit=1, verbose=False)

    def test_main_with_verbose(self, monkeypatch):
        """Test main function with verbose flag."""
        import backend.eval.answer.runner as runner_module

        mock_results = EvalResults(total=1, passed=1)
        mock_run_answer_eval = MagicMock(return_value=mock_results)

        monkeypatch.setattr(runner_module, "run_answer_eval", mock_run_answer_eval)
        monkeypatch.setattr(runner_module, "print_summary", MagicMock())

        runner_module.main(limit=5, verbose=True)

        call_kwargs = mock_run_answer_eval.call_args.kwargs
        assert call_kwargs["verbose"] is True
        assert call_kwargs["limit"] == 5
