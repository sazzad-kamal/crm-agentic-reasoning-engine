"""Tests for backend.eval.fetch module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.eval.fetch.models import CaseResult, EvalResults, Question

# =============================================================================
# Tests for models.py
# =============================================================================


class TestQuestion:
    """Tests for Question dataclass."""

    def test_question_basic(self):
        """Test basic Question creation."""
        q = Question(text="What is the stage?", difficulty=1)
        assert q.text == "What is the stage?"
        assert q.difficulty == 1

    def test_question_with_expected_sql(self):
        """Test Question with expected_sql field."""
        q = Question(
            text="What is the plan?",
            difficulty=1,
            expected_sql="SELECT plan FROM companies",
        )
        assert q.expected_sql == "SELECT plan FROM companies"

    def test_question_expected_sql_optional(self):
        """Test expected_sql is optional."""
        q = Question(text="Test", difficulty=1)
        assert q.expected_sql is None


class TestCaseResult:
    """Tests for CaseResult dataclass."""

    def test_case_result_basic(self):
        """Test basic CaseResult creation."""
        case = CaseResult(
            question=Question(text="Test question", difficulty=1),
            sql="SELECT * FROM companies",
            passed=True,
        )
        assert case.question.text == "Test question"
        assert case.question.difficulty == 1
        assert case.sql == "SELECT * FROM companies"
        assert case.passed is True
        assert case.errors == []

    def test_case_result_with_metrics(self):
        """Test CaseResult with all metrics."""
        case = CaseResult(
            question=Question(text="Test question", difficulty=3),
            sql="SELECT * FROM companies",
            passed=True,
            errors=[],
            latency_ms=150.0,
        )
        assert case.latency_ms == 150.0

    def test_case_result_with_errors(self):
        """Test CaseResult with errors."""
        case = CaseResult(
            question=Question(text="Test question", difficulty=2),
            sql="INVALID SQL",
            passed=False,
            errors=["SQL error: syntax error", "Validation failed"],
        )
        assert case.passed is False
        assert len(case.errors) == 2
        assert "SQL error" in case.errors[0]


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

    def test_compute_aggregates_latency(self):
        """Test compute_aggregates computes latency averages."""
        results = EvalResults()
        results.cases = [
            CaseResult(
                question=Question(text="Q1", difficulty=1),
                sql="SELECT 1",
                passed=True,
                latency_ms=150.0,
            ),
            CaseResult(
                question=Question(text="Q2", difficulty=1),
                sql="SELECT 2",
                passed=True,
                latency_ms=300.0,
            ),
        ]
        results.compute_aggregates()

        assert results.avg_latency_ms == 225.0


# =============================================================================
# Tests for sql_judge.py
# =============================================================================


class TestJudgeSqlEquivalence:
    """Tests for judge_sql_equivalence function."""

    def _mock_chain(self, result):
        """Create a mock chain that returns the given JudgeResult."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = result
        return mock_chain

    def test_judge_sql_equivalence_passed(self, monkeypatch):
        """Test successful judgment that passes."""
        import backend.eval.fetch.sql_judge as sql_judge_module
        from backend.eval.fetch.sql_judge import JudgeResult, judge_sql_equivalence

        mock_chain = self._mock_chain(JudgeResult(passed=True, errors=[]))
        monkeypatch.setattr(sql_judge_module, "create_openai_chain", lambda **kwargs: mock_chain)

        passed, errors = judge_sql_equivalence(
            generated_sql="SELECT COUNT(*) FROM companies",
            expected_sql="SELECT COUNT(*) FROM companies",
        )

        assert passed is True
        assert errors == []

    def test_judge_sql_equivalence_failed(self, monkeypatch):
        """Test judgment that fails."""
        import backend.eval.fetch.sql_judge as sql_judge_module
        from backend.eval.fetch.sql_judge import JudgeResult, judge_sql_equivalence

        mock_chain = self._mock_chain(JudgeResult(passed=False, errors=["Different table queried"]))
        monkeypatch.setattr(sql_judge_module, "create_openai_chain", lambda **kwargs: mock_chain)

        passed, errors = judge_sql_equivalence(
            generated_sql="SELECT COUNT(*) FROM contacts",
            expected_sql="SELECT COUNT(*) FROM companies",
        )

        assert passed is False
        assert "Different table queried" in errors

    def test_judge_sql_equivalence_api_error(self, monkeypatch):
        """Test API error returns failure."""
        import backend.eval.fetch.sql_judge as sql_judge_module
        from backend.eval.fetch.sql_judge import judge_sql_equivalence

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API connection error")
        monkeypatch.setattr(sql_judge_module, "create_openai_chain", lambda **kwargs: mock_chain)

        passed, errors = judge_sql_equivalence(
            generated_sql="SELECT 1",
            expected_sql="SELECT 1",
        )

        assert passed is False
        assert any("API error" in e for e in errors)

    def test_judge_sql_equivalence_empty_generated(self, monkeypatch):
        """Test handling when generated SQL is empty."""
        import backend.eval.fetch.sql_judge as sql_judge_module
        from backend.eval.fetch.sql_judge import JudgeResult, judge_sql_equivalence

        captured_args = {}

        def capture_invoke(inputs):
            captured_args.update(inputs)
            return JudgeResult(passed=False, errors=["No SQL generated"])

        mock_chain = MagicMock()
        mock_chain.invoke = capture_invoke
        monkeypatch.setattr(sql_judge_module, "create_openai_chain", lambda **kwargs: mock_chain)

        judge_sql_equivalence(
            generated_sql="",
            expected_sql="SELECT 1",
        )

        # Check that "No SQL generated" was used
        assert captured_args["generated_sql"] == "No SQL generated"


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
  - text: "Question 2"
    difficulty: 2
    expected_sql: "SELECT 2"
  - text: "Question 3"
    difficulty: 3
    expected_sql: "SELECT 3"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)

        import backend.eval.fetch.runner as runner_module

        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        from backend.eval.fetch.runner import load_questions

        questions = load_questions()
        assert len(questions) == 3
        assert questions[0].text == "Question 1"
        assert questions[0].expected_sql == "SELECT 1"

    def test_load_questions_with_expected_sql(self, monkeypatch, tmp_path):
        """Test loading questions with expected_sql."""
        yaml_content = """
questions:
  - text: "What is the plan?"
    difficulty: 1
    expected_sql: "SELECT plan FROM companies WHERE name = 'Acme'"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)

        import backend.eval.fetch.runner as runner_module

        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        from backend.eval.fetch.runner import load_questions

        questions = load_questions()
        assert questions[0].expected_sql == "SELECT plan FROM companies WHERE name = 'Acme'"


class TestRunSqlEval:
    """Tests for run_sql_eval function."""

    def test_run_sql_eval_basic(self, monkeypatch, tmp_path):
        """Test basic SQL evaluation run."""
        import backend.eval.fetch.runner as runner_module

        yaml_content = """
questions:
  - text: "Test question 1"
    difficulty: 1
    expected_sql: "SELECT 1"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        # Mock get_sql_plan
        mock_plan = MagicMock()
        mock_plan.sql = "SELECT 1"
        mock_get_sql_plan = MagicMock(return_value=mock_plan)
        monkeypatch.setattr(runner_module, "get_sql_plan", mock_get_sql_plan)

        # Mock get_connection
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(1,)]
        mock_result.description = [("value",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_get_conn = MagicMock(return_value=mock_conn)
        monkeypatch.setattr(runner_module, "get_connection", mock_get_conn)

        # Mock judge
        mock_judge = MagicMock(return_value=(True, []))
        monkeypatch.setattr(runner_module, "judge_sql_equivalence", mock_judge)

        from backend.eval.fetch.runner import run_sql_eval

        results = run_sql_eval(verbose=False)

        assert results.total == 1
        assert results.passed == 1
        assert len(results.cases) == 1
        assert results.cases[0].passed is True

    def test_run_sql_eval_with_limit(self, monkeypatch, tmp_path):
        """Test SQL evaluation with limit."""
        import backend.eval.fetch.runner as runner_module

        yaml_content = "questions:\n" + "\n".join(
            [f'  - text: "Question {i}"\n    difficulty: 1\n    expected_sql: "SELECT {i}"' for i in range(10)]
        )
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        mock_plan = MagicMock(sql="SELECT 1")
        monkeypatch.setattr(runner_module, "get_sql_plan", MagicMock(return_value=mock_plan))

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(1,)]
        mock_result.description = [("v",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))
        monkeypatch.setattr(runner_module, "judge_sql_equivalence", MagicMock(return_value=(True, [])))

        from backend.eval.fetch.runner import run_sql_eval

        results = run_sql_eval(limit=3)

        assert results.total == 3
        assert len(results.cases) == 3

    def test_run_sql_eval_planner_error(self, monkeypatch, tmp_path):
        """Test handling planner errors."""
        import backend.eval.fetch.runner as runner_module

        yaml_content = """
questions:
  - text: "Test"
    difficulty: 1
    expected_sql: "SELECT 1"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        mock_get_sql_plan = MagicMock(side_effect=Exception("Planner failed"))
        monkeypatch.setattr(runner_module, "get_sql_plan", mock_get_sql_plan)
        monkeypatch.setattr(runner_module, "get_connection", MagicMock())

        from backend.eval.fetch.runner import run_sql_eval

        results = run_sql_eval(verbose=False)

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].passed is False
        assert any("Planner error" in e for e in results.cases[0].errors)

    def test_run_sql_eval_sql_error(self, monkeypatch, tmp_path):
        """Test handling SQL execution errors."""
        import backend.eval.fetch.runner as runner_module

        yaml_content = """
questions:
  - text: "Test"
    difficulty: 1
    expected_sql: "SELECT 1"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        mock_plan = MagicMock(sql="INVALID SQL")
        monkeypatch.setattr(runner_module, "get_sql_plan", MagicMock(return_value=mock_plan))

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("SQL syntax error")
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))

        from backend.eval.fetch.runner import run_sql_eval

        results = run_sql_eval(verbose=False)

        assert results.total == 1
        assert results.passed == 0
        assert any("SQL error" in e for e in results.cases[0].errors)

    def test_run_sql_eval_uses_expected_sql(self, monkeypatch, tmp_path):
        """Test that eval passes expected_sql to judge."""
        import backend.eval.fetch.runner as runner_module

        yaml_content = """
questions:
  - text: "What is the plan?"
    difficulty: 1
    expected_sql: "SELECT plan FROM companies"
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        mock_plan = MagicMock(sql="SELECT plan FROM companies WHERE id = 1")
        monkeypatch.setattr(runner_module, "get_sql_plan", MagicMock(return_value=mock_plan))

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("Pro",)]
        mock_result.description = [("plan",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))

        # Capture judge call
        captured_calls = []

        def mock_judge(generated_sql, expected_sql):
            captured_calls.append({
                "generated_sql": generated_sql,
                "expected_sql": expected_sql,
            })
            return (True, [])

        monkeypatch.setattr(runner_module, "judge_sql_equivalence", mock_judge)

        from backend.eval.fetch.runner import run_sql_eval

        run_sql_eval(verbose=False)

        assert len(captured_calls) == 1
        assert captured_calls[0]["expected_sql"] == "SELECT plan FROM companies"
        assert captured_calls[0]["generated_sql"] == "SELECT plan FROM companies WHERE id = 1"


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary_basic(self, monkeypatch):
        """Test basic summary printing."""
        from backend.eval.fetch.runner import print_summary

        results = EvalResults(
            total=10,
            passed=9,
        )
        results.cases = [
            CaseResult(
                question=Question(text="Q", difficulty=1),
                sql="SELECT 1",
                passed=True,
                errors=[],
            )
            for _ in range(9)
        ]
        results.cases.append(
            CaseResult(
                question=Question(text="Q", difficulty=1),
                sql="SELECT 1",
                passed=False,
                errors=["Err"],
            )
        )
        results.compute_aggregates()

        # Should not raise
        print_summary(results)

    def test_print_summary_with_failures(self, monkeypatch):
        """Test summary printing with failed cases."""
        from backend.eval.fetch.runner import print_summary

        results = EvalResults(total=3, passed=1)
        results.cases = [
            CaseResult(
                question=Question(text="Passed question", difficulty=1),
                sql="SELECT 1",
                passed=True,
                errors=[],
            ),
            CaseResult(
                question=Question(text="Failed question 1", difficulty=2),
                sql="SELECT bad",
                passed=False,
                errors=["SQL error"],
            ),
            CaseResult(
                question=Question(text="Failed question 2", difficulty=3),
                sql="SELECT worse",
                passed=False,
                errors=["Another error"],
            ),
        ]
        results.compute_aggregates()

        # Should not raise
        print_summary(results)

    def test_print_summary_many_failures(self, monkeypatch):
        """Test summary printing with more than 10 failures (truncation)."""
        from backend.eval.fetch.runner import print_summary

        results = EvalResults(total=15, passed=0)
        results.cases = [
            CaseResult(
                question=Question(text=f"Failed question {i}", difficulty=1),
                sql=f"SELECT {i}",
                passed=False,
                errors=[f"Error {i}"],
            )
            for i in range(15)
        ]
        results.compute_aggregates()

        # Should not raise, should show "... and X more failures"
        print_summary(results)


# =============================================================================
# Tests for __main__.py
# =============================================================================


class TestMainCli:
    """Tests for CLI main function."""

    def test_main_basic(self, monkeypatch):
        """Test main function runs without error."""
        import backend.eval.fetch.runner as runner_module

        mock_results = EvalResults(total=1, passed=1)

        monkeypatch.setattr(runner_module, "run_sql_eval", MagicMock(return_value=mock_results))
        monkeypatch.setattr(runner_module, "print_summary", MagicMock())

        # Should not raise
        runner_module.main(limit=1, verbose=False)

    def test_main_with_verbose(self, monkeypatch):
        """Test main function with verbose flag."""
        import backend.eval.fetch.runner as runner_module

        mock_results = EvalResults(total=1, passed=1)
        mock_run_sql_eval = MagicMock(return_value=mock_results)

        monkeypatch.setattr(runner_module, "run_sql_eval", mock_run_sql_eval)
        monkeypatch.setattr(runner_module, "print_summary", MagicMock())

        runner_module.main(limit=5, verbose=True)

        call_kwargs = mock_run_sql_eval.call_args.kwargs
        assert call_kwargs["verbose"] is True
        assert call_kwargs["limit"] == 5
