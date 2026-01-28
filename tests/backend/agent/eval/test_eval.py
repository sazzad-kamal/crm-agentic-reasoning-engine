"""
Tests for backend.eval module.

Tests the evaluation models and shared utilities.
"""

import json
import os
from typing import Any

import pytest

# Set mock mode
os.environ["MOCK_LLM"] = "1"


def make_invoke_mock(result: dict[str, Any]) -> dict[str, Any]:
    """Helper to create _invoke_agent return value."""
    return result


from backend.eval.integration.models import (
    SLO_FLOW_PASS_RATE,
    FlowEvalResults,
    FlowResult,
    FlowStepResult,
)

# =============================================================================
# Flow Model Tests
# =============================================================================


class TestFlowStepResult:
    """Tests for FlowStepResult."""

    def test_creation(self):
        result = FlowStepResult(
            question="What is Acme's revenue?",
            answer="Acme's revenue is $1M.",
            latency_ms=500,
            has_answer=True,
            relevance_score=0.9,
        )
        assert result.relevance_score == 0.9

    def test_passed_property(self):
        passing = FlowStepResult(
            question="Q", answer="A", latency_ms=100, has_answer=True,
            relevance_score=0.8, ragas_metrics_total=2,
        )
        assert passing.passed is True

        failing_relevance = FlowStepResult(
            question="Q", answer="A", latency_ms=100, has_answer=True,
            relevance_score=0.5, ragas_metrics_total=2,
        )
        assert failing_relevance.passed is False

        failing_no_answer = FlowStepResult(
            question="Q", answer="", latency_ms=100, has_answer=False,
            relevance_score=0.8, ragas_metrics_total=2,
        )
        assert failing_no_answer.passed is False

        passing_no_ragas = FlowStepResult(
            question="Q", answer="A", latency_ms=100, has_answer=True,
            relevance_score=0.0, ragas_metrics_total=0,
        )
        assert passing_no_ragas.passed is True

    def test_passed_with_errors(self):
        result = FlowStepResult(
            question="Q", answer="A", latency_ms=100, has_answer=True,
            relevance_score=0.9, errors=["Something went wrong"],
        )
        assert result.passed is False

    def test_low_relevance_fails(self):
        result = FlowStepResult(
            question="Q", answer="A", latency_ms=100, has_answer=True,
            relevance_score=0.5, ragas_metrics_total=2,
        )
        assert result.passed is False


class TestFlowResult:
    """Tests for FlowResult."""

    def test_creation(self):
        steps = [
            FlowStepResult(question="Q1", answer="A1", latency_ms=100, has_answer=True, relevance_score=0.9),
            FlowStepResult(question="Q2", answer="A2", latency_ms=150, has_answer=True, relevance_score=0.85),
        ]
        result = FlowResult(
            path_id=1, questions=["Q1", "Q2"], steps=steps,
            total_latency_ms=250, success=True,
        )
        assert result.path_id == 1
        assert len(result.steps) == 2
        assert result.success is True


class TestFlowEvalResults:
    """Tests for FlowEvalResults (extends BaseEvalResults)."""

    def test_creation(self):
        results = FlowEvalResults(
            total=10, passed=8,
            avg_relevance=0.85,
            total_latency_ms=5000,
            avg_latency_per_question_ms=166.7,
        )
        assert results.passed == 8
        assert results.avg_relevance == 0.85

    def test_pass_rate(self):
        results = FlowEvalResults(total=10, passed=8)
        assert results.pass_rate == 0.8

    def test_failed_property(self):
        results = FlowEvalResults(total=10, passed=8)
        assert results.failed == 2

    def test_empty(self):
        results = FlowEvalResults(total=0, passed=0)
        assert results.pass_rate == 0.0

    def test_compute_aggregates(self):
        cases = [
            FlowResult(
                path_id=0, questions=["Q1?"], success=True, total_latency_ms=200,
                steps=[FlowStepResult(
                    question="Q1?", answer="A1", latency_ms=200, has_answer=True,
                    relevance_score=0.9, answer_correctness_score=0.8,
                    ragas_metrics_total=2, ragas_metrics_failed=0,
                )],
            ),
            FlowResult(
                path_id=1, questions=["Q2?"], success=False, total_latency_ms=100,
                steps=[FlowStepResult(
                    question="Q2?", answer="A2", latency_ms=100, has_answer=True,
                    relevance_score=0.5, answer_correctness_score=0.3,
                    ragas_metrics_total=2, ragas_metrics_failed=1,
                )],
            ),
        ]
        results = FlowEvalResults(total=2, cases=cases)
        results.compute_aggregates()

        assert results.passed == 1
        assert results.failed == 1
        assert results.avg_relevance == 0.7
        assert results.avg_answer_correctness == 0.55
        assert results.ragas_metrics_total == 4
        assert results.ragas_metrics_failed == 1
        assert results.total_latency_ms == 300


# =============================================================================
# SLO Constants Tests
# =============================================================================


class TestSLOConstants:
    """Tests for SLO constant values."""

    def test_slo_flow_pass_rate(self):
        assert SLO_FLOW_PASS_RATE == 0.85


# =============================================================================
# RAGAS Success Rate Tests
# =============================================================================


class TestModelsExtended:
    """Extended tests for models module."""

    def test_ragas_success_rate_zero_total(self):
        results = FlowEvalResults(total=1, passed=1, ragas_metrics_total=0, ragas_metrics_failed=0)
        assert results.ragas_success_rate == 1.0

    def test_ragas_success_rate_all_failed(self):
        results = FlowEvalResults(total=1, passed=0, ragas_metrics_total=5, ragas_metrics_failed=5)
        assert results.ragas_success_rate == 0.0

    def test_ragas_success_rate_partial_success(self):
        results = FlowEvalResults(total=1, passed=1, ragas_metrics_total=10, ragas_metrics_failed=3)
        assert results.ragas_success_rate == 0.7


# =============================================================================
# LangSmith Latency Tests (with mocks)
# =============================================================================


class TestLangSmithLatency:
    """Tests for LangSmith latency percentages with mocks."""

    def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        from backend.eval.integration.langsmith import get_latency_percentages
        assert get_latency_percentages() == {}

    def test_no_runs(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        class MockClient:
            def list_runs(self, **kwargs): return []
        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)
        from backend.eval.integration.langsmith import get_latency_percentages
        assert get_latency_percentages() == {}

    def test_with_agent_runs(self, monkeypatch):
        from datetime import datetime, timedelta
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")

        class MockRun:
            def __init__(self, name, start, end):
                self.name = name
                self.start_time = start
                self.end_time = end

        now = datetime.utcnow()
        class MockClient:
            def list_runs(self, **kwargs):
                return [
                    MockRun("fetch", now, now + timedelta(milliseconds=400)),
                    MockRun("answer", now, now + timedelta(milliseconds=300)),
                    MockRun("followup", now, now + timedelta(milliseconds=200)),
                    MockRun("route", now, now + timedelta(milliseconds=100)),  # non-agent, filtered
                ]

        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)
        from backend.eval.integration.langsmith import get_latency_percentages
        result = get_latency_percentages()
        assert abs(result["fetch"] - 400 / 900) < 0.01
        assert abs(result["answer"] - 300 / 900) < 0.01
        assert abs(result["followup"] - 200 / 900) < 0.01

    def test_api_error(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        class MockClient:
            def list_runs(self, **kwargs): raise Exception("API Error")
        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)
        from backend.eval.integration.langsmith import get_latency_percentages
        assert get_latency_percentages() == {}


# =============================================================================
# Output Module Tests
# =============================================================================


class TestOutputModule:
    """Tests for print_summary function."""

    def test_print_summary_all_pass(self):
        from backend.eval.integration.runner import print_summary
        results = FlowEvalResults(
            total=10, passed=9,
            avg_relevance=0.90, avg_answer_correctness=0.75,
            avg_latency_per_question_ms=3000,
            ragas_metrics_total=100, ragas_metrics_failed=5,
        )
        result = print_summary(results)
        assert isinstance(result, bool)

    def test_print_summary_with_latency_pcts(self):
        from backend.eval.integration.runner import print_summary
        results = FlowEvalResults(
            total=5, passed=4,
            avg_relevance=0.90, avg_answer_correctness=0.72,
        )
        result = print_summary(results, latency_pcts={"fetch": 0.20, "answer": 0.30})
        assert isinstance(result, bool)

    def test_print_summary_no_latency_pcts(self):
        from backend.eval.integration.runner import print_summary
        results = FlowEvalResults(total=5, passed=4, avg_relevance=0.90)
        result = print_summary(results)
        assert isinstance(result, bool)

    def test_print_summary_with_failed_paths(self):
        from backend.eval.integration.runner import print_summary
        cases = [
            FlowResult(
                path_id=0, questions=["Q1?"], success=False, total_latency_ms=100,
                steps=[FlowStepResult(
                    question="Q1?", answer="", latency_ms=100, has_answer=False,
                    errors=["Agent crashed"],
                )],
            ),
        ]
        results = FlowEvalResults(total=1, cases=cases)
        results.compute_aggregates()
        result = print_summary(results)
        assert result is False


# =============================================================================
# Runner Module Tests
# =============================================================================


class TestTestSingleQuestion:
    """Tests for test_single_question function."""

    def test_success(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "This is a good answer with enough content.",
                "sql_results": {"company_info": [{"name": "Acme", "company_id": "ACME001"}]},
            })
        def mock_evaluate_single(*args, **kwargs):
            return {"answer_relevancy": 0.90, "answer_correctness": 0.70, "nan_metrics": []}

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: "Expected")

        result = test_single_question("What is Acme's status?", "session1")
        assert result.has_answer is True
        assert result.relevance_score == 0.90

    def test_no_answer(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({"answer": "", "sql_results": {}})

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: None)

        result = test_single_question("Hello?", "session1", use_judge=False)
        assert result.has_answer is False
        assert result.relevance_score == 0.0

    def test_exception(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            raise RuntimeError("Agent crashed")

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)

        result = test_single_question("Q?", "session1")
        assert result.has_answer is False
        assert len(result.errors) > 0
        assert "Agent crashed" in result.errors[0]

    def test_with_context(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Good answer with sufficient length.",
                "sql_results": {"data": [{"id": 1}]},
            })
        def mock_evaluate_single(*args, **kwargs):
            return {"answer_relevancy": 0.85, "answer_correctness": 0.75, "nan_metrics": []}

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: "Expected")

        result = test_single_question("Q?", "session1")
        assert result.relevance_score == 0.85

    def test_with_sql_results(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Test answer with good content",
                "sql_results": {"query": [{"company": "Acme", "value": 100}]},
            })
        def mock_evaluate_single(*args, **kwargs):
            return {"answer_relevancy": 0.90, "answer_correctness": 0.75, "nan_metrics": []}

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: "Expected")

        result = test_single_question("Test Q?", "session1")
        assert result.relevance_score == 0.90

    def test_ragas_failed(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Test answer with good content",
                "sql_results": {"data": [{"id": 1}]},
            })
        def mock_evaluate_single(*args, **kwargs):
            return {"answer_relevancy": 0.0, "answer_correctness": 0.0, "nan_metrics": ["answer_relevancy", "answer_correctness"]}

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: "Expected")

        result = test_single_question("Q?", "session1")
        assert result.ragas_metrics_failed >= 2

    def test_ragas_exception(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Test answer with good content",
                "sql_results": {"data": [{"id": 1}]},
            })
        def mock_evaluate_single(*args, **kwargs):
            raise RuntimeError("RAGAS evaluation crashed")

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: "Expected")

        result = test_single_question("Q?", "session1")
        assert result.has_answer is True
        assert result.relevance_score == 0.0  # RAGAS failed, defaults to 0

    def test_no_expected_answer(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Test answer with content",
                "sql_results": {"query": [{"a": 1}]},
            })
        def mock_evaluate_single(*args, **kwargs):
            return {"answer_relevancy": 0.85, "answer_correctness": 0.0, "nan_metrics": []}

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: None)

        result = test_single_question("Q?", "session1")
        assert result.relevance_score == 0.85


class TestTestFlow:
    """Tests for test_flow function."""

    def test_success(self, monkeypatch):
        from backend.eval.integration.runner import test_flow
        call_count = {"count": 0}
        def mock_test_single_question(question, session_id, use_judge=True):
            call_count["count"] += 1
            return FlowStepResult(
                question=question, answer=f"Answer to {question}",
                latency_ms=100, has_answer=True, relevance_score=0.90,
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "test_single_question", mock_test_single_question)

        result = test_flow(["Q1?", "Q2?"], path_id=0)
        assert result.success is True
        assert len(result.steps) == 2
        assert result.total_latency_ms == 200
        assert call_count["count"] == 2

    def test_with_failure(self, monkeypatch):
        from backend.eval.integration.runner import test_flow
        def mock_test_single_question(question, session_id, use_judge=True):
            if "fail" in question.lower():
                return FlowStepResult(
                    question=question, answer="", latency_ms=100,
                    has_answer=False, relevance_score=0.0,
                    errors=["Failed to answer"],
                )
            return FlowStepResult(
                question=question, answer="Good answer", latency_ms=100,
                has_answer=True, relevance_score=0.90,
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "test_single_question", mock_test_single_question)

        result = test_flow(["Q1?", "Q2 fail?"], path_id=0)
        assert result.success is False
        step_errors = [e for s in result.steps for e in s.errors]
        assert "Failed to answer" in step_errors


# =============================================================================
# Runner Module Tests (run_flow_eval)
# =============================================================================


class TestRunFlowEval:
    """Tests for run_flow_eval function."""

    def test_basic(self, monkeypatch):
        from backend.eval.integration.runner import run_flow_eval
        def mock_get_all_paths(): return [["Q1?", "Q2?"]]
        def mock_test_flow(questions, path_id, use_judge=True):
            return FlowResult(
                path_id=path_id, questions=questions,
                steps=[
                    FlowStepResult(question="Q1?", answer="A1", latency_ms=100, has_answer=True, relevance_score=0.9),
                    FlowStepResult(question="Q2?", answer="A2", latency_ms=100, has_answer=True, relevance_score=0.85),
                ],
                total_latency_ms=200, success=True,
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1)
        assert results.total == 1
        assert results.passed == 1

    def test_with_failures(self, monkeypatch):
        from backend.eval.integration.runner import run_flow_eval
        def mock_get_all_paths(): return [["Q1?"]]
        def mock_test_flow(questions, path_id, use_judge=True):
            return FlowResult(
                path_id=path_id, questions=questions,
                steps=[FlowStepResult(question="Q1?", answer="Bad", latency_ms=100, has_answer=True, relevance_score=0.5)],
                total_latency_ms=100, success=False,
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1)
        assert results.failed == 1
        assert len([c for c in results.cases if not c.success]) == 1

    def test_with_answer_correctness(self, monkeypatch):
        from backend.eval.integration.runner import run_flow_eval
        def mock_get_all_paths(): return [["Q1?"]]
        def mock_test_flow(questions, path_id, use_judge=True):
            return FlowResult(
                path_id=path_id, questions=questions,
                steps=[FlowStepResult(
                    question="Q1?", answer="A1", latency_ms=100, has_answer=True,
                    relevance_score=0.9, answer_correctness_score=0.85,
                )],
                total_latency_ms=100, success=True,
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1)
        assert results.avg_answer_correctness == 0.85

    def test_multiple_paths(self, monkeypatch):
        from backend.eval.integration.runner import run_flow_eval
        def mock_get_all_paths(): return [["Q1?"], ["Q2?"], ["Q3?"]]
        call_count = {"count": 0}
        def mock_test_flow(questions, path_id, use_judge=True):
            call_count["count"] += 1
            return FlowResult(
                path_id=path_id, questions=questions,
                steps=[FlowStepResult(
                    question=questions[0], answer=f"A{path_id}", latency_ms=100,
                    has_answer=True, relevance_score=0.9,
                )],
                total_latency_ms=100, success=True,
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=3)
        assert results.total == 3
        assert call_count["count"] == 3


# =============================================================================
# LangSmith Tests (Extended)
# =============================================================================


class TestLangSmithPercentages:
    """Tests for get_latency_percentages function."""

    def test_success(self, monkeypatch):
        from datetime import datetime, timedelta
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        class MockRun:
            def __init__(self, name, start, end):
                self.name, self.start_time, self.end_time = name, start, end
        now = datetime.utcnow()
        class MockClient:
            def list_runs(self, **kwargs):
                return [
                    MockRun("fetch", now, now + timedelta(milliseconds=400)),
                    MockRun("answer", now, now + timedelta(milliseconds=300)),
                    MockRun("followup", now, now + timedelta(milliseconds=200)),
                ]
        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)
        from backend.eval.integration.langsmith import get_latency_percentages
        result = get_latency_percentages()
        assert abs(result["fetch"] - 400 / 900) < 0.01
        assert abs(result["answer"] - 300 / 900) < 0.01

    def test_empty_breakdown(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        class MockClient:
            def list_runs(self, **kwargs): return []
        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)
        from backend.eval.integration.langsmith import get_latency_percentages
        assert get_latency_percentages() == {}

    def test_no_agent_nodes(self, monkeypatch):
        from datetime import datetime, timedelta
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        class MockRun:
            def __init__(self, name, start, end):
                self.name, self.start_time, self.end_time = name, start, end
        now = datetime.utcnow()
        class MockClient:
            def list_runs(self, **kwargs):
                return [MockRun("other", now, now + timedelta(milliseconds=100))]
        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)
        from backend.eval.integration.langsmith import get_latency_percentages
        assert get_latency_percentages() == {}

    def test_import_error(self, monkeypatch):
        import sys
        monkeypatch.delitem(sys.modules, "langsmith", raising=False)
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "langsmith": raise ImportError("No module named 'langsmith'")
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        from backend.eval.integration.langsmith import get_latency_percentages
        assert get_latency_percentages() == {}

    def test_zero_total(self, monkeypatch):
        from datetime import datetime
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")
        class MockRun:
            def __init__(self, name, start, end):
                self.name, self.start_time, self.end_time = name, start, end
        now = datetime.utcnow()
        class MockClient:
            def list_runs(self, **kwargs):
                return [MockRun("route", now, now), MockRun("answer", now, now)]
        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)
        from backend.eval.integration.langsmith import get_latency_percentages
        assert get_latency_percentages() == {}


# =============================================================================
# Tree Module Tests
# =============================================================================


class TestTreeValidation:
    def test_get_tree_stats(self):
        from backend.eval.integration.tree import get_tree_stats
        stats = get_tree_stats()
        for key in ("num_starters", "num_questions", "num_edges", "num_paths", "max_depth", "path_lengths"):
            assert key in stats

    def test_get_all_paths(self):
        from backend.eval.integration.tree import get_all_paths
        paths = get_all_paths()
        assert isinstance(paths, list)
        if paths:
            assert isinstance(paths[0], list)


class TestYamlLoading:
    def test_get_expected_answer_exists(self):
        from backend.eval.integration.tree import get_expected_answer
        result = get_expected_answer("What deals are in the pipeline?")
        assert result is None or isinstance(result, str)

    def test_get_expected_answer_not_exists(self):
        from backend.eval.integration.tree import get_expected_answer
        assert get_expected_answer("This question does not exist") is None


class TestYamlLoadingErrors:
    def test_nonexistent_file(self, monkeypatch, tmp_path):
        import backend.eval.integration.tree as tree_module
        tree_module._load_expected_answers.cache_clear()
        monkeypatch.setattr(tree_module, "_EVAL_FIXTURES_PATH", tmp_path / "nonexistent")
        assert tree_module._load_expected_answers() == {}
        tree_module._load_expected_answers.cache_clear()

    def test_invalid_yaml(self, monkeypatch, tmp_path):
        import backend.eval.integration.tree as tree_module
        tree_module._load_expected_answers.cache_clear()
        (tmp_path / "expected_answers.yaml").write_text("::invalid:: yaml: [content")
        monkeypatch.setattr(tree_module, "_EVAL_FIXTURES_PATH", tmp_path)
        assert tree_module._load_expected_answers() == {}
        tree_module._load_expected_answers.cache_clear()

    def test_empty_file(self, monkeypatch, tmp_path):
        import backend.eval.integration.tree as tree_module
        tree_module._load_expected_answers.cache_clear()
        (tmp_path / "expected_answers.yaml").write_text("")
        monkeypatch.setattr(tree_module, "_EVAL_FIXTURES_PATH", tmp_path)
        assert tree_module._load_expected_answers() == {}
        tree_module._load_expected_answers.cache_clear()


class TestTreePathFinding:
    def test_compute_max_depth_no_descendants(self):
        import networkx as nx
        from backend.eval.integration.tree import _compute_max_depth
        mock_g = nx.DiGraph()
        mock_g.add_node("Starter Q?")
        assert _compute_max_depth(mock_g, ["Starter Q?"]) == 0

    def test_find_paths_with_nx_no_path(self):
        import networkx as nx
        from backend.eval.integration.tree import _find_paths
        mock_g = nx.DiGraph()
        mock_g.add_edge("Starter?", "Child?")
        mock_g.add_edge("Child?", "Leaf?")
        result = _find_paths(mock_g, ["Starter?"], 5)
        assert len(result) > 0


class TestTreeNetworkPaths:
    def test_compute_max_depth_no_path_between_nodes(self):
        import networkx as nx
        from backend.eval.integration.tree import _compute_max_depth
        G = nx.DiGraph()
        G.add_edge("starter", "node1")
        G.add_node("orphan")
        assert _compute_max_depth(G, ["starter"]) >= 1

    def test_find_paths_disconnected_nodes(self):
        import networkx as nx
        from backend.eval.integration.tree import _find_paths
        G = nx.DiGraph()
        G.add_edge("starter", "mid")
        G.add_edge("mid", "leaf")
        assert len(_find_paths(G, ["starter"], 3)) >= 1


# =============================================================================
# CLI Module Tests
# =============================================================================


class TestCliModuleExtended:
    def test_fetch_command(self, monkeypatch):
        from backend.eval.fetch.runner import main as fetch_main
        from dataclasses import dataclass

        @dataclass
        class MockResults:
            total: int = 10
            passed: int = 9
            failed: int = 1
            pass_rate: float = 0.9
            cases: list = None
            def __post_init__(self): self.cases = []

        import backend.eval.fetch.runner
        monkeypatch.setattr(backend.eval.fetch.runner, "run_sql_eval", lambda **kw: MockResults())
        monkeypatch.setattr(backend.eval.fetch.runner, "print_summary", lambda r: None)
        fetch_main(limit=1, verbose=False)

    def test_main_command(self, monkeypatch):
        from backend.eval.integration.runner import main
        call_args = {}
        def mock_run_eval(**kwargs): call_args.update(kwargs)
        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_run_eval", mock_run_eval)
        main(limit=5)
        assert call_args["limit"] == 5


# =============================================================================
# Judge / Suppression Tests
# =============================================================================


class TestJudgeModule:
    def test_suppress_event_loop_closed_errors(self):
        from backend.eval.answer.text.suppression import install_event_loop_error_suppression
        install_event_loop_error_suppression()


# =============================================================================
# RAGAS Tests
# =============================================================================


class TestRagasSuppression:
    def test_already_run(self):
        from backend.eval.answer.text.suppression import install_event_loop_error_suppression
        install_event_loop_error_suppression()

    def test_event_loop_closed_filter(self):
        import logging
        from backend.eval.answer.text import ragas  # noqa: F401
        asyncio_logger = logging.getLogger("asyncio")
        assert any(hasattr(f, "filter") for f in asyncio_logger.filters)

    def test_ragas_executor_filter(self):
        import logging
        executor_logger = logging.getLogger("ragas.executor")
        assert any(hasattr(f, "filter") for f in executor_logger.filters)


class TestRagasEvaluateSingle:
    @pytest.mark.no_mock_llm
    def test_empty_contexts(self):
        from unittest.mock import MagicMock, patch
        import pandas as pd
        from backend.eval.answer.text import ragas
        df = pd.DataFrame({"answer_correctness": [0.75], "answer_relevancy": [0.80]})
        mock_eval_result = MagicMock()
        mock_eval_result.to_pandas.return_value = df
        with patch.object(ragas, "evaluate", return_value=mock_eval_result):
            with patch.object(ragas, "_evaluators", return_value=(MagicMock(), MagicMock())):
                result = ragas.evaluate_single("Q?", "Paris", [], "Paris is the capital")
        assert "answer_correctness" in result

    @pytest.mark.no_mock_llm
    def test_nan_metrics(self):
        from unittest.mock import MagicMock, patch
        import pandas as pd
        from backend.eval.answer.text import ragas
        df = pd.DataFrame({"answer_correctness": [float("nan")], "answer_relevancy": [float("nan")]})
        mock_eval_result = MagicMock()
        mock_eval_result.to_pandas.return_value = df
        with patch.object(ragas, "evaluate", return_value=mock_eval_result):
            with patch.object(ragas, "_evaluators", return_value=(MagicMock(), MagicMock())):
                result = ragas.evaluate_single("Test?", "Answer", ["Context"], "Expected")
        assert "answer_correctness" in result["nan_metrics"]
        assert result["answer_correctness"] == 0.0

    @pytest.mark.no_mock_llm
    def test_evaluators_returns_two_metrics(self):
        from unittest.mock import MagicMock, patch
        from backend.eval.answer.text import ragas
        with patch.object(ragas, "get_langchain_chat_openai", return_value=MagicMock()):
            ragas._evaluators.cache_clear()
            metrics = ragas._evaluators()
        assert len(metrics) == 2
        ragas._evaluators.cache_clear()

    @pytest.mark.no_mock_llm
    def test_extract_scores_with_nan_values(self):
        from unittest.mock import MagicMock
        import pandas as pd
        from backend.eval.answer.text import ragas
        df = pd.DataFrame({"answer_correctness": [None], "answer_relevancy": [None]})
        mock_eval_result = MagicMock()
        mock_eval_result.to_pandas.return_value = df
        result = ragas._extract_scores(mock_eval_result)
        assert result["answer_correctness"] == 0.0
        assert "answer_correctness" in result["nan_metrics"]


class TestMainMiddleware:
    @pytest.mark.asyncio
    @pytest.mark.no_mock_llm
    async def test_request_logging_middleware(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from backend.main import RequestLoggingMiddleware
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        @app.get("/test")
        async def test_endpoint(): return {"status": "ok"}
        client = TestClient(app)
        assert client.get("/test").status_code == 200
