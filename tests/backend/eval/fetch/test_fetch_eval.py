"""Tests for backend.eval.fetch module."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch

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
        assert q.rag_only is False

    def test_question_rag_only(self):
        """Test Question with rag_only=True."""
        q = Question(text="What are the notes?", difficulty=2, rag_only=True)
        assert q.text == "What are the notes?"
        assert q.difficulty == 2
        assert q.rag_only is True


class TestCaseResult:
    """Tests for CaseResult dataclass."""

    def test_case_result_basic(self):
        """Test basic CaseResult creation."""
        case = CaseResult(
            question="Test question",
            difficulty=1,
            rag_only=False,
            sql="SELECT * FROM companies",
            passed=True,
        )
        assert case.question == "Test question"
        assert case.difficulty == 1
        assert case.rag_only is False
        assert case.sql == "SELECT * FROM companies"
        assert case.passed is True
        assert case.row_count == 0
        assert case.errors == []

    def test_case_result_with_metrics(self):
        """Test CaseResult with all metrics."""
        case = CaseResult(
            question="Test question",
            difficulty=3,
            rag_only=False,
            sql="SELECT * FROM companies",
            passed=True,
            row_count=5,
            errors=[],
            sql_gen_latency_ms=100.0,
            sql_exec_latency_ms=50.0,
            rag_latency_ms=200.0,
            total_latency_ms=350.0,
            rag_precision=0.85,
            rag_recall=0.75,
        )
        assert case.row_count == 5
        assert case.sql_gen_latency_ms == 100.0
        assert case.sql_exec_latency_ms == 50.0
        assert case.rag_latency_ms == 200.0
        assert case.total_latency_ms == 350.0
        assert case.rag_precision == 0.85
        assert case.rag_recall == 0.75

    def test_case_result_with_errors(self):
        """Test CaseResult with errors."""
        case = CaseResult(
            question="Test question",
            difficulty=2,
            rag_only=False,
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
        assert results.sql_executed == 0
        assert results.sql_failed == 0
        assert results.rag_invoked == 0
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

    def test_sql_correctness_property(self):
        """Test sql_correctness property."""
        results = EvalResults()
        results.cases = [
            CaseResult(question="Q1", difficulty=1, rag_only=False, sql="SELECT 1", passed=True, errors=[]),
            CaseResult(question="Q2", difficulty=1, rag_only=False, sql="SELECT 2", passed=True, errors=[]),
            CaseResult(question="Q3", difficulty=1, rag_only=False, sql="SELECT 3", passed=False, errors=["Error"]),
        ]
        # 2 out of 3 SQL questions passed
        assert results.sql_correctness == pytest.approx(2 / 3)

    def test_sql_correctness_with_rag_only(self):
        """Test sql_correctness excludes rag_only questions."""
        results = EvalResults()
        results.cases = [
            CaseResult(question="Q1", difficulty=1, rag_only=False, sql="SELECT 1", passed=True, errors=[]),
            CaseResult(question="Q2", difficulty=1, rag_only=True, sql="", passed=True, errors=[]),
            CaseResult(question="Q3", difficulty=1, rag_only=False, sql="SELECT 3", passed=False, errors=["Err"]),
        ]
        # Only 2 SQL questions, 1 passed
        assert results.sql_correctness == 0.5

    def test_sql_correctness_no_sql_questions(self):
        """Test sql_correctness with no SQL questions."""
        results = EvalResults()
        results.cases = [
            CaseResult(question="Q1", difficulty=1, rag_only=True, sql="", passed=True, errors=[]),
        ]
        assert results.sql_correctness == 0.0

    def test_compute_aggregates_empty_cases(self):
        """Test compute_aggregates with empty cases."""
        results = EvalResults()
        results.compute_aggregates()
        # Should not raise, values stay at defaults
        assert results.avg_sql_gen_latency_ms == 0.0

    def test_compute_aggregates_latency(self):
        """Test compute_aggregates computes latency averages."""
        results = EvalResults()
        results.cases = [
            CaseResult(
                question="Q1",
                difficulty=1,
                rag_only=False,
                sql="SELECT 1",
                passed=True,
                sql_gen_latency_ms=100.0,
                sql_exec_latency_ms=50.0,
                total_latency_ms=150.0,
            ),
            CaseResult(
                question="Q2",
                difficulty=1,
                rag_only=False,
                sql="SELECT 2",
                passed=True,
                sql_gen_latency_ms=200.0,
                sql_exec_latency_ms=100.0,
                total_latency_ms=300.0,
            ),
        ]
        results.compute_aggregates()

        assert results.avg_sql_gen_latency_ms == 150.0
        assert results.avg_sql_exec_latency_ms == 75.0
        assert results.avg_total_latency_ms == 225.0

    def test_compute_aggregates_rag_metrics(self):
        """Test compute_aggregates computes RAG averages."""
        results = EvalResults()
        results.cases = [
            CaseResult(
                question="Q1",
                difficulty=1,
                rag_only=False,
                sql="SELECT 1",
                passed=True,
                rag_latency_ms=100.0,
                rag_precision=0.9,
                rag_recall=0.8,
            ),
            CaseResult(
                question="Q2",
                difficulty=1,
                rag_only=False,
                sql="SELECT 2",
                passed=True,
                rag_latency_ms=200.0,
                rag_precision=0.7,
                rag_recall=0.6,
            ),
            CaseResult(
                question="Q3",
                difficulty=1,
                rag_only=False,
                sql="SELECT 3",
                passed=True,
                rag_latency_ms=0.0,  # No RAG
                rag_precision=None,
                rag_recall=None,
            ),
        ]
        results.compute_aggregates()

        # RAG latency average: (100 + 200) / 2 = 150
        assert results.avg_rag_latency_ms == 150.0
        assert results.rag_invoked == 2
        # Precision average: (0.9 + 0.7) / 2 = 0.8
        assert results.avg_rag_precision == 0.8
        # Recall average: (0.8 + 0.6) / 2 = 0.7
        assert results.avg_rag_recall == 0.7


# =============================================================================
# Tests for sql_judge.py
# =============================================================================


class TestFormatResults:
    """Tests for _format_results function."""

    def test_format_results_empty(self):
        """Test formatting empty results."""
        from backend.eval.fetch.sql_judge import _format_results

        result = _format_results({})
        assert result == "No results returned"

    def test_format_results_none(self):
        """Test formatting None results."""
        from backend.eval.fetch.sql_judge import _format_results

        result = _format_results(None)
        assert result == "No results returned"

    def test_format_results_normal(self):
        """Test formatting normal results."""
        from backend.eval.fetch.sql_judge import _format_results

        data = {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}
        result = _format_results(data)
        assert "column1" in result
        assert "column2" in result

    def test_format_results_truncation(self):
        """Test results are truncated when too large."""
        from backend.eval.fetch.sql_judge import _format_results

        # Create large data that exceeds 4000 chars
        data = {"column": ["x" * 100] * 100}
        result = _format_results(data)
        assert "... (truncated)" in result
        assert len(result) <= 4100  # 4000 + truncation message


class TestGetOpenaiClient:
    """Tests for get_openai_client singleton."""

    def test_get_openai_client_singleton(self, monkeypatch):
        """Test client is created as singleton via lru_cache."""
        from backend.core.llm import get_openai_client

        # Clear cache to test fresh creation
        get_openai_client.cache_clear()

        mock_client = MagicMock()

        with patch("openai.OpenAI", return_value=mock_client):
            client1 = get_openai_client()
            client2 = get_openai_client()

            # Should be same instance (cached)
            assert client1 is client2

        get_openai_client.cache_clear()


class TestJudgeSqlResults:
    """Tests for judge_sql_results function."""

    def test_judge_sql_results_passed(self, monkeypatch):
        """Test successful judgment that passes."""
        import backend.eval.fetch.sql_judge as sql_judge_module

        # call_openai_json returns dict directly
        monkeypatch.setattr(
            sql_judge_module,
            "call_openai_json",
            lambda prompt: {"passed": True, "reasoning": "Good", "errors": []},
        )

        from backend.eval.fetch.sql_judge import judge_sql_results

        passed, errors = judge_sql_results(
            question="What is the count?",
            sql="SELECT COUNT(*) FROM companies",
            sql_results={"count": [5]},
        )

        assert passed is True
        assert errors == []

    def test_judge_sql_results_failed(self, monkeypatch):
        """Test judgment that fails."""
        import backend.eval.fetch.sql_judge as sql_judge_module

        monkeypatch.setattr(
            sql_judge_module,
            "call_openai_json",
            lambda prompt: {"passed": False, "reasoning": "Wrong count", "errors": ["Count mismatch"]},
        )

        from backend.eval.fetch.sql_judge import judge_sql_results

        passed, errors = judge_sql_results(
            question="What is the count?",
            sql="SELECT COUNT(*) FROM companies",
            sql_results={"count": [5]},
        )

        assert passed is False
        assert "Count mismatch" in errors

    def test_judge_sql_results_failed_with_reasoning_no_errors(self, monkeypatch):
        """Test judgment failed with reasoning but no errors list."""
        import backend.eval.fetch.sql_judge as sql_judge_module

        monkeypatch.setattr(
            sql_judge_module,
            "call_openai_json",
            lambda prompt: {"passed": False, "reasoning": "Data is incomplete", "errors": []},
        )

        from backend.eval.fetch.sql_judge import judge_sql_results

        passed, errors = judge_sql_results(
            question="What is the count?",
            sql="SELECT COUNT(*) FROM companies",
            sql_results={"count": [5]},
        )

        assert passed is False
        # Reasoning should be used as error when errors list is empty
        assert "Data is incomplete" in errors

    def test_judge_sql_results_invalid_json(self, monkeypatch):
        """Test handling of invalid JSON response."""
        import backend.eval.fetch.sql_judge as sql_judge_module

        def raise_json_error(prompt):
            raise json.JSONDecodeError("Invalid JSON", "", 0)

        monkeypatch.setattr(sql_judge_module, "call_openai_json", raise_json_error)

        from backend.eval.fetch.sql_judge import judge_sql_results

        passed, errors = judge_sql_results(
            question="Test",
            sql="SELECT 1",
            sql_results={"val": [1]},
        )

        assert passed is False
        assert "Judge failed to return valid JSON" in errors

    def test_judge_sql_results_api_error_with_retry(self, monkeypatch):
        """Test API error triggers retry and eventually fails."""
        import backend.eval.fetch.sql_judge as sql_judge_module

        call_count = {"count": 0}

        def raise_api_error(prompt):
            call_count["count"] += 1
            raise Exception("API connection error")

        monkeypatch.setattr(sql_judge_module, "call_openai_json", raise_api_error)

        from backend.eval.fetch.sql_judge import judge_sql_results

        passed, errors = judge_sql_results(
            question="Test",
            sql="SELECT 1",
            sql_results={"val": [1]},
        )

        assert passed is False
        assert any("API error" in e for e in errors)
        # Should have tried twice
        assert call_count["count"] == 2

    def test_judge_sql_results_retry_succeeds(self, monkeypatch):
        """Test retry succeeds after first failure."""
        import backend.eval.fetch.sql_judge as sql_judge_module

        call_count = {"count": 0}

        def retry_success(prompt):
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise Exception("Temporary error")
            return {"passed": True, "reasoning": "Good", "errors": []}

        monkeypatch.setattr(sql_judge_module, "call_openai_json", retry_success)

        from backend.eval.fetch.sql_judge import judge_sql_results

        passed, errors = judge_sql_results(
            question="Test",
            sql="SELECT 1",
            sql_results={"val": [1]},
        )

        assert passed is True
        assert errors == []
        assert call_count["count"] == 2

    def test_judge_sql_results_no_sql(self, monkeypatch):
        """Test handling when SQL is empty."""
        import backend.eval.fetch.sql_judge as sql_judge_module

        captured_prompt = {}

        def capture_prompt(prompt):
            captured_prompt["prompt"] = prompt
            return {"passed": True, "reasoning": "OK", "errors": []}

        monkeypatch.setattr(sql_judge_module, "call_openai_json", capture_prompt)

        from backend.eval.fetch.sql_judge import judge_sql_results

        judge_sql_results(
            question="Test",
            sql="",
            sql_results={"val": [1]},
        )

        # Check that "No SQL provided" was used in prompt
        assert "No SQL provided" in captured_prompt["prompt"]

    def test_judge_sql_results_errors_not_list(self, monkeypatch):
        """Test handling when errors is not a list."""
        import backend.eval.fetch.sql_judge as sql_judge_module

        monkeypatch.setattr(
            sql_judge_module,
            "call_openai_json",
            lambda prompt: {"passed": False, "reasoning": "Bad", "errors": "single error"},
        )

        from backend.eval.fetch.sql_judge import judge_sql_results

        passed, errors = judge_sql_results(
            question="Test",
            sql="SELECT 1",
            sql_results={"val": [1]},
        )

        assert passed is False
        # Should convert string to list
        assert isinstance(errors, list)
        assert "single error" in errors


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
  - text: "Question 2"
    difficulty: 2
    rag_only: true
  - text: "Question 3"
    difficulty: 3
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)

        import backend.eval.fetch.runner as runner_module

        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        from backend.eval.fetch.runner import load_questions

        questions = load_questions()
        assert len(questions) == 3
        assert questions[0].text == "Question 1"
        assert questions[1].rag_only is True

    def test_load_questions_difficulty_filter(self, monkeypatch, tmp_path):
        """Test filtering by difficulty."""
        yaml_content = """
questions:
  - text: "Easy"
    difficulty: 1
  - text: "Medium"
    difficulty: 3
  - text: "Hard"
    difficulty: 5
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)

        import backend.eval.fetch.runner as runner_module

        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        from backend.eval.fetch.runner import load_questions

        questions = load_questions(difficulty_filter=[1, 5])
        assert len(questions) == 2
        assert questions[0].difficulty == 1
        assert questions[1].difficulty == 5

    def test_load_questions_rag_only_true(self, monkeypatch, tmp_path):
        """Test filtering for rag_only=True."""
        yaml_content = """
questions:
  - text: "SQL Question"
    difficulty: 1
  - text: "RAG Question"
    difficulty: 1
    rag_only: true
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)

        import backend.eval.fetch.runner as runner_module

        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        from backend.eval.fetch.runner import load_questions

        questions = load_questions(rag_only_filter=True)
        assert len(questions) == 1
        assert questions[0].rag_only is True

    def test_load_questions_rag_only_false(self, monkeypatch, tmp_path):
        """Test filtering for rag_only=False (SQL only)."""
        yaml_content = """
questions:
  - text: "SQL Question"
    difficulty: 1
  - text: "RAG Question"
    difficulty: 1
    rag_only: true
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)

        import backend.eval.fetch.runner as runner_module

        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        from backend.eval.fetch.runner import load_questions

        questions = load_questions(rag_only_filter=False)
        assert len(questions) == 1
        assert questions[0].rag_only is False


class TestEvaluateRag:
    """Tests for _evaluate_rag function."""

    def test_evaluate_rag_no_data(self):
        """Test RAG evaluation with no data."""
        from backend.eval.fetch.runner import _evaluate_rag

        results = EvalResults()
        latency, precision, recall = _evaluate_rag("Test question", [], results, False)

        assert latency == 0.0
        assert precision is None
        assert recall is None

    def test_evaluate_rag_no_entity_ids(self):
        """Test RAG evaluation with no entity IDs in data."""
        from backend.eval.fetch.runner import _evaluate_rag

        results = EvalResults()
        data = [{"name": "Test", "value": 123}]  # No company_id, contact_id, etc.
        latency, precision, recall = _evaluate_rag("Test question", data, results, False)

        assert latency == 0.0
        assert precision is None
        assert recall is None

    def test_evaluate_rag_with_entity_ids(self, monkeypatch):
        """Test RAG evaluation with valid entity IDs."""
        import backend.eval.fetch.runner as runner_module

        mock_search = MagicMock(return_value=("Some context\n\n---\n\nMore context", []))
        monkeypatch.setattr(runner_module, "search_entity_context", mock_search)

        mock_evaluate = MagicMock(return_value={"context_precision": 0.9, "context_recall": 0.8})
        monkeypatch.setattr(runner_module, "evaluate_single", mock_evaluate)

        from backend.eval.fetch.runner import _evaluate_rag

        results = EvalResults()
        # Note: keys must exactly match what the function looks for
        data = [{"company_id": "ACME-001", "name": "Acme"}]
        latency, precision, recall = _evaluate_rag("Test question", data, results, False)

        # Verify the search was called with correct entity_ids
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[0][0] == "Test question"
        assert "company_id" in call_args[0][1]

        assert latency >= 0  # Latency can be very small in tests
        assert precision == 0.9
        assert recall == 0.8
        assert results.rag_invoked == 1

    def test_evaluate_rag_empty_context(self, monkeypatch):
        """Test RAG evaluation with empty context returned."""
        import backend.eval.fetch.runner as runner_module

        mock_search = MagicMock(return_value=("", []))
        monkeypatch.setattr(runner_module, "search_entity_context", mock_search)

        from backend.eval.fetch.runner import _evaluate_rag

        results = EvalResults()
        data = [{"company_id": "ACME-001"}]
        latency, precision, recall = _evaluate_rag("Test question", data, results, False)

        # Verify search was called
        mock_search.assert_called_once()

        # Should still record latency but no precision/recall without evaluate_single call
        assert latency >= 0  # Can be very small in tests
        assert precision is None
        assert recall is None
        assert results.rag_invoked == 1

    def test_evaluate_rag_exception(self, monkeypatch):
        """Test RAG evaluation handles exceptions."""
        import backend.eval.fetch.runner as runner_module

        mock_search = MagicMock(side_effect=Exception("RAG error"))
        monkeypatch.setattr(runner_module, "search_entity_context", mock_search)

        from backend.eval.fetch.runner import _evaluate_rag

        results = EvalResults()
        data = [{"company_id": "ACME-001"}]
        latency, precision, recall = _evaluate_rag("Test question", data, results, verbose=False)

        # Should return latency but None for metrics
        assert latency >= 0
        assert precision is None
        assert recall is None


class TestRunSqlEval:
    """Tests for run_sql_eval function."""

    def test_run_sql_eval_basic(self, monkeypatch):
        """Test basic SQL evaluation run."""
        import backend.eval.fetch.runner as runner_module

        questions = [
            Question(text="Test question 1", difficulty=1, rag_only=False),
        ]

        # Mock get_sql_plan
        mock_plan = MagicMock()
        mock_plan.sql = "SELECT 1"
        mock_plan.needs_rag = False
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
        monkeypatch.setattr(runner_module, "judge_sql_results", mock_judge)

        from backend.eval.fetch.runner import run_sql_eval

        results = run_sql_eval(questions=questions, verbose=False)

        assert results.total == 1
        assert results.passed == 1
        assert results.sql_executed == 1
        assert len(results.cases) == 1
        assert results.cases[0].passed is True

    def test_run_sql_eval_with_limit(self, monkeypatch):
        """Test SQL evaluation with limit."""
        import backend.eval.fetch.runner as runner_module

        questions = [
            Question(text=f"Question {i}", difficulty=1) for i in range(10)
        ]

        mock_plan = MagicMock(sql="SELECT 1", needs_rag=False)
        monkeypatch.setattr(runner_module, "get_sql_plan", MagicMock(return_value=mock_plan))

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(1,)]
        mock_result.description = [("v",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))
        monkeypatch.setattr(runner_module, "judge_sql_results", MagicMock(return_value=(True, [])))

        from backend.eval.fetch.runner import run_sql_eval

        results = run_sql_eval(questions=questions, limit=3)

        assert results.total == 3
        assert len(results.cases) == 3

    def test_run_sql_eval_planner_error(self, monkeypatch):
        """Test handling planner errors."""
        import backend.eval.fetch.runner as runner_module

        questions = [Question(text="Test", difficulty=1)]

        mock_get_sql_plan = MagicMock(side_effect=Exception("Planner failed"))
        monkeypatch.setattr(runner_module, "get_sql_plan", mock_get_sql_plan)
        monkeypatch.setattr(runner_module, "get_connection", MagicMock())

        from backend.eval.fetch.runner import run_sql_eval

        results = run_sql_eval(questions=questions, verbose=False)

        assert results.total == 1
        assert results.passed == 0
        assert results.cases[0].passed is False
        assert any("Planner error" in e for e in results.cases[0].errors)

    def test_run_sql_eval_sql_error(self, monkeypatch):
        """Test handling SQL execution errors."""
        import backend.eval.fetch.runner as runner_module

        questions = [Question(text="Test", difficulty=1)]

        mock_plan = MagicMock(sql="INVALID SQL", needs_rag=False)
        monkeypatch.setattr(runner_module, "get_sql_plan", MagicMock(return_value=mock_plan))

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("SQL syntax error")
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))

        from backend.eval.fetch.runner import run_sql_eval

        results = run_sql_eval(questions=questions, verbose=False)

        assert results.total == 1
        assert results.passed == 0
        assert results.sql_failed == 1
        assert any("SQL error" in e for e in results.cases[0].errors)

    def test_run_sql_eval_with_rag(self, monkeypatch):
        """Test SQL evaluation that triggers RAG."""
        import backend.eval.fetch.runner as runner_module

        questions = [Question(text="Test", difficulty=1)]

        mock_plan = MagicMock(sql="SELECT * FROM companies", needs_rag=True)
        monkeypatch.setattr(runner_module, "get_sql_plan", MagicMock(return_value=mock_plan))

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("ACME-001", "Acme")]
        mock_result.description = [("company_id",), ("name",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))
        monkeypatch.setattr(runner_module, "judge_sql_results", MagicMock(return_value=(True, [])))

        # Mock _evaluate_rag
        mock_evaluate_rag = MagicMock(return_value=(100.0, 0.9, 0.8))
        monkeypatch.setattr(runner_module, "_evaluate_rag", mock_evaluate_rag)

        from backend.eval.fetch.runner import run_sql_eval

        results = run_sql_eval(questions=questions, verbose=False)

        assert results.passed == 1
        assert results.cases[0].rag_latency_ms == 100.0
        assert results.cases[0].rag_precision == 0.9
        assert results.cases[0].rag_recall == 0.8

    def test_run_sql_eval_loads_questions(self, monkeypatch, tmp_path):
        """Test run_sql_eval loads questions when not provided."""
        import backend.eval.fetch.runner as runner_module

        yaml_content = """
questions:
  - text: "Auto loaded question"
    difficulty: 1
"""
        yaml_file = tmp_path / "questions.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setattr(runner_module, "QUESTIONS_PATH", yaml_file)

        mock_plan = MagicMock(sql="SELECT 1", needs_rag=False)
        monkeypatch.setattr(runner_module, "get_sql_plan", MagicMock(return_value=mock_plan))

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(1,)]
        mock_result.description = [("v",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))
        monkeypatch.setattr(runner_module, "judge_sql_results", MagicMock(return_value=(True, [])))

        from backend.eval.fetch.runner import run_sql_eval

        results = run_sql_eval(questions=None, difficulty_filter=[1])

        assert results.total == 1
        assert results.cases[0].question == "Auto loaded question"

    def test_run_sql_eval_verbose_output(self, monkeypatch, capsys):
        """Test verbose output during evaluation."""
        import backend.eval.fetch.runner as runner_module

        questions = [Question(text="Verbose test", difficulty=2)]

        mock_plan = MagicMock(sql="SELECT 1", needs_rag=False)
        monkeypatch.setattr(runner_module, "get_sql_plan", MagicMock(return_value=mock_plan))

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(1,)]
        mock_result.description = [("v",)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        monkeypatch.setattr(runner_module, "get_connection", MagicMock(return_value=mock_conn))
        monkeypatch.setattr(runner_module, "judge_sql_results", MagicMock(return_value=(True, [])))

        from backend.eval.fetch.runner import run_sql_eval

        run_sql_eval(questions=questions, verbose=True)
        # The function uses rich console, so we just verify it doesn't error with verbose=True


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary_basic(self, monkeypatch):
        """Test basic summary printing."""
        from backend.eval.fetch.runner import print_summary

        results = EvalResults(
            total=10,
            passed=9,
            sql_executed=10,
            sql_failed=0,
            rag_invoked=0,
        )
        results.cases = [
            CaseResult(question="Q", difficulty=1, rag_only=False, sql="SELECT 1", passed=True, errors=[])
            for _ in range(9)
        ]
        results.cases.append(
            CaseResult(question="Q", difficulty=1, rag_only=False, sql="SELECT 1", passed=False, errors=["Err"])
        )
        results.compute_aggregates()

        # Should not raise
        print_summary(results)

    def test_print_summary_with_rag(self, monkeypatch):
        """Test summary printing with RAG metrics."""
        from backend.eval.fetch.runner import print_summary

        results = EvalResults(
            total=5,
            passed=4,
            sql_executed=5,
            sql_failed=0,
            rag_invoked=3,
            avg_rag_precision=0.85,
            avg_rag_recall=0.75,
            avg_rag_latency_ms=150.0,
        )
        results.cases = [
            CaseResult(
                question="Q",
                difficulty=1,
                rag_only=False,
                sql="SELECT 1",
                passed=True,
                rag_latency_ms=150.0,
            )
            for _ in range(5)
        ]

        # Should not raise
        print_summary(results)

    def test_print_summary_with_failures(self, monkeypatch):
        """Test summary printing with failed cases."""
        from backend.eval.fetch.runner import print_summary

        results = EvalResults(total=3, passed=1)
        results.cases = [
            CaseResult(
                question="Passed question",
                difficulty=1,
                rag_only=False,
                sql="SELECT 1",
                passed=True,
                errors=[],
            ),
            CaseResult(
                question="Failed question 1",
                difficulty=2,
                rag_only=False,
                sql="SELECT bad",
                passed=False,
                errors=["SQL error"],
            ),
            CaseResult(
                question="Failed question 2",
                difficulty=3,
                rag_only=False,
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
                question=f"Failed question {i}",
                difficulty=1,
                rag_only=False,
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
        import backend.eval.fetch.__main__ as main_module

        mock_results = EvalResults(total=1, passed=1)

        monkeypatch.setattr(main_module, "run_sql_eval", MagicMock(return_value=mock_results))
        monkeypatch.setattr(main_module, "print_summary", MagicMock())

        # Should not raise
        main_module.main(limit=1, verbose=False, difficulty=None)

    def test_main_with_difficulty_filter(self, monkeypatch):
        """Test main function with difficulty filter."""
        import backend.eval.fetch.__main__ as main_module

        mock_results = EvalResults(total=1, passed=1)
        mock_run_sql_eval = MagicMock(return_value=mock_results)

        monkeypatch.setattr(main_module, "run_sql_eval", mock_run_sql_eval)
        monkeypatch.setattr(main_module, "print_summary", MagicMock())

        main_module.main(limit=None, verbose=False, difficulty="1,3,5")

        # Verify difficulty filter was parsed correctly
        call_kwargs = mock_run_sql_eval.call_args.kwargs
        assert call_kwargs["difficulty_filter"] == [1, 3, 5]

    def test_main_with_single_difficulty(self, monkeypatch):
        """Test main function with single difficulty."""
        import backend.eval.fetch.__main__ as main_module

        mock_results = EvalResults(total=1, passed=1)
        mock_run_sql_eval = MagicMock(return_value=mock_results)

        monkeypatch.setattr(main_module, "run_sql_eval", mock_run_sql_eval)
        monkeypatch.setattr(main_module, "print_summary", MagicMock())

        main_module.main(limit=None, verbose=True, difficulty="2")

        call_kwargs = mock_run_sql_eval.call_args.kwargs
        assert call_kwargs["difficulty_filter"] == [2]

    def test_main_invalid_difficulty(self, monkeypatch):
        """Test main function with invalid difficulty filter."""
        from backend.eval.fetch.__main__ import main

        import typer

        with pytest.raises(typer.Exit):
            main(limit=None, verbose=False, difficulty="invalid")

    def test_main_with_verbose(self, monkeypatch):
        """Test main function with verbose flag."""
        import backend.eval.fetch.__main__ as main_module

        mock_results = EvalResults(total=1, passed=1)
        mock_run_sql_eval = MagicMock(return_value=mock_results)

        monkeypatch.setattr(main_module, "run_sql_eval", mock_run_sql_eval)
        monkeypatch.setattr(main_module, "print_summary", MagicMock())

        main_module.main(limit=5, verbose=True, difficulty=None)

        call_kwargs = mock_run_sql_eval.call_args.kwargs
        assert call_kwargs["verbose"] is True
        assert call_kwargs["limit"] == 5
