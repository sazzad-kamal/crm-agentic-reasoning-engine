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
    """Helper to create _invoke_agent return value.

    Args:
        result: The workflow state result dict

    Returns:
        Result dict as expected by _invoke_agent
    """
    return result


from backend.eval.integration.models import (
    SLO_FLOW_PATH_PASS_RATE,
    FlowEvalResults,
    FlowResult,
    FlowStepResult,
)

# =============================================================================
# Flow Model Tests
# =============================================================================


class TestFlowStepResult:
    """Tests for FlowStepResult dataclass."""

    def test_flow_step_result_creation(self):
        """Test creating a FlowStepResult."""
        result = FlowStepResult(
            question="What is Acme's revenue?",
            answer="Acme's revenue is $1M.",
            latency_ms=500,
            has_answer=True,
            relevance_score=0.9,

        )

        assert result.relevance_score == 0.9

    def test_flow_step_result_passed_property(self):
        """Test FlowStepResult.passed property."""
        # Passing case
        passing = FlowStepResult(
            question="Q",
            answer="A",
            latency_ms=100,
            has_answer=True,
            relevance_score=0.8,

        )
        assert passing.passed is True

        # Failing - low relevance
        failing_relevance = FlowStepResult(
            question="Q",
            answer="A",
            latency_ms=100,
            has_answer=True,
            relevance_score=0.5,

        )
        assert failing_relevance.passed is False

        # Failing - no answer
        failing_no_answer = FlowStepResult(
            question="Q",
            answer="",
            latency_ms=100,
            has_answer=False,
            relevance_score=0.8,

        )
        assert failing_no_answer.passed is False


class TestFlowResult:
    """Tests for FlowResult dataclass."""

    def test_flow_result_creation(self):
        """Test creating a FlowResult."""
        steps = [
            FlowStepResult(
                question="Q1",
                answer="A1",
                latency_ms=100,
                has_answer=True,
                    relevance_score=0.9,
    
            ),
            FlowStepResult(
                question="Q2",
                answer="A2",
                latency_ms=150,
                has_answer=True,
                    relevance_score=0.85,
    
            ),
        ]

        result = FlowResult(
            path_id=1,
            questions=["Q1", "Q2"],
            steps=steps,
            total_latency_ms=250,
            success=True,
        )

        assert result.path_id == 1
        assert len(result.steps) == 2
        assert result.success is True


class TestFlowEvalResults:
    """Tests for FlowEvalResults dataclass."""

    def test_flow_eval_results_creation(self):
        """Test creating FlowEvalResults."""
        results = FlowEvalResults(
            total_paths=10,
            paths_tested=10,
            paths_passed=8,
            paths_failed=2,
            total_questions=30,
            questions_passed=27,
            questions_failed=3,
            avg_relevance=0.85,

            total_latency_ms=5000,
            avg_latency_per_question_ms=166.7,
        )

        assert results.paths_passed == 8
        assert results.avg_relevance == 0.85

    def test_flow_eval_results_pass_rate_properties(self):
        """Test FlowEvalResults pass rate properties."""
        results = FlowEvalResults(
            total_paths=10,
            paths_tested=10,
            paths_passed=8,
            paths_failed=2,
            total_questions=30,
            questions_passed=27,
            questions_failed=3,
        )

        assert results.path_pass_rate == 0.8

    def test_flow_eval_results_empty(self):
        """Test FlowEvalResults with empty data."""
        results = FlowEvalResults(
            total_paths=0,
            paths_tested=0,
            paths_passed=0,
            paths_failed=0,
            total_questions=0,
            questions_passed=0,
            questions_failed=0,
        )

        assert results.path_pass_rate == 0.0


# =============================================================================
# SLO Constants Tests
# =============================================================================


class TestSLOConstants:
    """Tests for SLO constant values."""

    def test_slo_flow_path_pass_rate(self):
        """Test SLO flow path pass rate threshold."""
        assert SLO_FLOW_PATH_PASS_RATE == 0.85


# =============================================================================
# LangSmith Latency Tests (with mocks)
# =============================================================================


class TestLangSmithLatency:
    """Tests for LangSmith latency breakdown with mocks."""

    def test_get_latency_breakdown_no_api_key(self, monkeypatch):
        """Test get_latency_breakdown without API key."""
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        from backend.eval.integration.langsmith import get_latency_breakdown

        result = get_latency_breakdown()
        assert result == {}

    def test_get_latency_breakdown_no_runs(self, monkeypatch):
        """Test get_latency_breakdown with no runs found."""
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")

        class MockClient:
            def list_runs(self, **kwargs):
                return []

        # Mock at the langsmith module level since it's imported inside the function
        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)

        from backend.eval.integration.langsmith import get_latency_breakdown

        result = get_latency_breakdown()
        assert result == {}

    def test_get_latency_breakdown_with_runs(self, monkeypatch):
        """Test get_latency_breakdown with mock runs."""
        from datetime import datetime, timedelta

        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")

        class MockRun:
            def __init__(self, name, start, end, run_id="run1"):
                self.name = name
                self.start_time = start
                self.end_time = end
                self.id = run_id

        now = datetime.utcnow()

        class MockClient:
            def __init__(self):
                self.call_count = 0

            def list_runs(self, **kwargs):
                self.call_count += 1
                # is_root=False means child runs (agent nodes)
                if kwargs.get("is_root") is False:
                    return [
                        MockRun("route", now, now + timedelta(milliseconds=100)),
                        MockRun("fetch_rag", now, now + timedelta(milliseconds=500)),
                        MockRun("answer", now, now + timedelta(milliseconds=300)),
                    ]
                else:
                    # Parent runs
                    return [MockRun("agent", now, now + timedelta(seconds=1))]

        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)

        from backend.eval.integration.langsmith import get_latency_breakdown

        result = get_latency_breakdown()

        assert "route" in result
        assert "fetch_rag" in result
        assert "answer" in result
        assert result["route"]["avg_ms"] == 100.0
        assert result["fetch_rag"]["avg_ms"] == 500.0
        assert result["answer"]["avg_ms"] == 300.0

    def test_get_latency_breakdown_api_error(self, monkeypatch):
        """Test get_latency_breakdown handles API errors."""
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")

        class MockClient:
            def list_runs(self, **kwargs):
                raise Exception("API Error")

        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)

        from backend.eval.integration.langsmith import get_latency_breakdown

        result = get_latency_breakdown()
        assert result == {}


# =============================================================================
# Output Module Tests
# =============================================================================


class TestOutputModule:
    """Tests for print_summary function."""

    def test_print_summary_all_pass(self):
        """Test print_summary with all SLOs passing."""
        from backend.eval.integration.runner import print_summary

        results = FlowEvalResults(
            total_paths=10,
            paths_tested=10,
            paths_passed=9,
            paths_failed=1,
            total_questions=30,
            questions_passed=28,
            questions_failed=2,
            avg_relevance=0.90,

            avg_answer_correctness=0.75,
            avg_latency_per_question_ms=3000,
            ragas_metrics_total=100,
            ragas_metrics_failed=5,
        )

        # Should not raise and should return a boolean
        result = print_summary(results)
        assert isinstance(result, bool)

    def test_print_summary_with_latency_pcts(self):
        """Test print_summary with latency percentages."""
        from backend.eval.integration.runner import print_summary

        results = FlowEvalResults(
            total_paths=5,
            paths_tested=5,
            paths_passed=4,
            paths_failed=1,
            total_questions=15,
            questions_passed=13,
            questions_failed=2,
            avg_relevance=0.90,

            avg_answer_correctness=0.72,
        )

        latency_pcts = {"fetch": 0.20, "answer": 0.30}
        result = print_summary(results, latency_pcts=latency_pcts)
        assert isinstance(result, bool)

    def test_print_summary_no_latency_pcts(self):
        """Test print_summary without latency percentages."""
        from backend.eval.integration.runner import print_summary

        results = FlowEvalResults(
            total_paths=5,
            paths_tested=5,
            paths_passed=4,
            paths_failed=1,
            total_questions=15,
            questions_passed=13,
            questions_failed=2,
            avg_relevance=0.90,

        )

        result = print_summary(results)
        assert isinstance(result, bool)


# =============================================================================
# Runner Module Tests
# =============================================================================


class TestTestSingleQuestion:
    """Tests for test_single_question function."""

    def test_test_single_question_success(self, monkeypatch):
        """Test test_single_question with successful execution."""
        from backend.eval.integration.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "This is a good answer with enough content.",
                "sql_results": {"company_info": [{"name": "Acme", "company_id": "ACME001"}]},
            })

        def mock_get_expected_answer(question):
            return "Expected answer"

        def mock_evaluate_single(*args, **kwargs):
            return {
                "answer_relevancy": 0.90,
                "answer_correctness": 0.70,
                "nan_metrics": [],
            }

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("What is Acme's status?", "session1")

        assert result.has_answer is True
        assert result.relevance_score == 0.90

    def test_test_single_question_no_answer(self, monkeypatch):
        """Test test_single_question when agent returns no answer."""
        from backend.eval.integration.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({"answer": "", "sql_results": {}})

        def mock_get_expected_answer(question):
            return None

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Hello?", "session1", use_judge=False)

        assert result.has_answer is False
        assert result.relevance_score == 0.0

    def test_test_single_question_exception(self, monkeypatch):
        """Test test_single_question handles exceptions."""
        from backend.eval.integration.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            raise RuntimeError("Agent crashed")

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)

        result = test_single_question("Q?", "session1")

        assert result.has_answer is False
        assert result.error is not None
        assert "Agent crashed" in result.error

    def test_test_single_question_with_context(self, monkeypatch):
        """Test test_single_question with context from SQL."""
        from backend.eval.integration.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Good answer with sufficient length.",
                "sql_results": {"data": [{"id": 1}]},
            })

        def mock_get_expected_answer(question):
            return "Expected"

        def mock_evaluate_single(*args, **kwargs):
            return {
                "answer_relevancy": 0.85,
                "answer_correctness": 0.75,
                "nan_metrics": [],
            }

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Q?", "session1")

        assert result.relevance_score == 0.85


class TestTestFlow:
    """Tests for test_flow function."""

    def test_test_flow_success(self, monkeypatch):
        """Test test_flow with successful execution."""
        from backend.eval.integration.runner import test_flow

        call_count = {"count": 0}

        def mock_test_single_question(question, session_id, use_judge=True):
            call_count["count"] += 1
            return FlowStepResult(
                question=question,
                answer=f"Answer to {question}",
                latency_ms=100,
                has_answer=True,
                relevance_score=0.90,
    
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "test_single_question", mock_test_single_question)

        result = test_flow(["Q1?", "Q2?"], path_id=0)

        assert result.success is True
        assert len(result.steps) == 2
        assert result.total_latency_ms == 200
        assert call_count["count"] == 2

    def test_test_flow_with_failure(self, monkeypatch):
        """Test test_flow when a step fails."""
        from backend.eval.integration.runner import test_flow

        def mock_test_single_question(question, session_id, use_judge=True):
            if "fail" in question.lower():
                return FlowStepResult(
                    question=question,
                    answer="",
                    latency_ms=100,
                    has_answer=False,
                    relevance_score=0.0,

                    error="Failed to answer",
                )
            return FlowStepResult(
                question=question,
                answer="Good answer",
                latency_ms=100,
                has_answer=True,
                relevance_score=0.90,
    
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "test_single_question", mock_test_single_question)

        result = test_flow(["Q1?", "Q2 fail?"], path_id=0)

        assert result.success is False
        assert result.error == "Failed to answer"


# =============================================================================
# LangSmith Tests (Extended)
# =============================================================================


class TestLangSmithPercentages:
    """Tests for get_latency_percentages function."""

    def test_get_latency_percentages_success(self, monkeypatch):
        """Test get_latency_percentages with valid data."""
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
                ]

        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)

        from backend.eval.integration.langsmith import get_latency_percentages

        result = get_latency_percentages()

        assert "fetch" in result
        assert "answer" in result
        assert "followup" in result
        # Total avg = 400 + 300 + 200 = 900ms
        assert abs(result["fetch"] - 400 / 900) < 0.01
        assert abs(result["answer"] - 300 / 900) < 0.01

    def test_get_latency_percentages_empty_breakdown(self, monkeypatch):
        """Test get_latency_percentages with empty breakdown."""
        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key")

        class MockClient:
            def list_runs(self, **kwargs):
                return []

        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)

        from backend.eval.integration.langsmith import get_latency_percentages

        result = get_latency_percentages()
        assert result == {}

    def test_get_latency_percentages_no_agent_nodes(self, monkeypatch):
        """Test get_latency_percentages when no agent nodes found."""
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
                # Return runs that aren't agent nodes
                return [
                    MockRun("some_other_node", now, now + timedelta(milliseconds=100)),
                    MockRun("random_step", now, now + timedelta(milliseconds=200)),
                ]

        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)

        from backend.eval.integration.langsmith import get_latency_percentages

        result = get_latency_percentages()
        assert result == {}

    def test_get_latency_breakdown_import_error(self, monkeypatch):
        """Test get_latency_breakdown when langsmith not installed."""
        import sys
        # Remove langsmith from modules if present
        monkeypatch.delitem(sys.modules, "langsmith", raising=False)

        # Make import fail
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "langsmith":
                raise ImportError("No module named 'langsmith'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Need to reimport to trigger the import error
        # This is tricky - the function catches the import error internally
        from backend.eval.integration.langsmith import get_latency_breakdown
        result = get_latency_breakdown()
        assert result == {}


# =============================================================================
# Models Tests (Extended)
# =============================================================================


class TestModelsExtended:
    """Extended tests for models module."""

    def test_ragas_success_rate_zero_total(self):
        """Test ragas_success_rate when no metrics evaluated."""
        results = FlowEvalResults(
            total_paths=1,
            paths_tested=1,
            paths_passed=1,
            paths_failed=0,
            total_questions=1,
            questions_passed=1,
            questions_failed=0,
            ragas_metrics_total=0,  # No metrics evaluated
            ragas_metrics_failed=0,
        )

        assert results.ragas_success_rate == 1.0  # Default to 100% when no metrics

    def test_ragas_success_rate_all_failed(self):
        """Test ragas_success_rate when all metrics failed."""
        results = FlowEvalResults(
            total_paths=1,
            paths_tested=1,
            paths_passed=0,
            paths_failed=1,
            total_questions=1,
            questions_passed=0,
            questions_failed=1,
            ragas_metrics_total=5,
            ragas_metrics_failed=5,  # All failed
        )

        assert results.ragas_success_rate == 0.0

    def test_ragas_success_rate_partial_success(self):
        """Test ragas_success_rate with partial success."""
        results = FlowEvalResults(
            total_paths=1,
            paths_tested=1,
            paths_passed=1,
            paths_failed=0,
            total_questions=1,
            questions_passed=1,
            questions_failed=0,
            ragas_metrics_total=10,
            ragas_metrics_failed=3,  # 7 succeeded
        )

        assert results.ragas_success_rate == 0.7

    def test_flow_step_result_low_relevance_fails(self):
        """Test FlowStepResult.passed with low relevance."""
        result = FlowStepResult(
            question="Q",
            answer="A",
            latency_ms=100,
            has_answer=True,
            relevance_score=0.5,  # Below 0.7 threshold
        )

        assert result.passed is False


# =============================================================================
# Runner Module Tests (Extended - run_flow_eval)
# =============================================================================


class TestRunFlowEval:
    """Tests for run_flow_eval function."""

    def test_run_flow_eval_basic(self, monkeypatch):
        """Test run_flow_eval with basic execution."""
        from backend.eval.integration.runner import run_flow_eval

        def mock_get_all_paths():
            return [["Q1?", "Q2?"]]

        def mock_test_flow(questions, path_id, use_judge=True):
            return FlowResult(
                path_id=path_id,
                questions=questions,
                steps=[
                    FlowStepResult(
                        question="Q1?",
                        answer="A1",
                        latency_ms=100,
                        has_answer=True,
                        relevance_score=0.9,
            
                    ),
                    FlowStepResult(
                        question="Q2?",
                        answer="A2",
                        latency_ms=100,
                        has_answer=True,
                        relevance_score=0.85,
            
                    ),
                ],
                total_latency_ms=200,
                success=True,
            )

        import backend.eval.integration.runner

        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1)

        assert results.total_paths == 1
        assert results.paths_passed == 1
        assert results.total_questions == 2

    def test_run_flow_eval_with_failures(self, monkeypatch):
        """Test run_flow_eval with failed paths."""
        from backend.eval.integration.runner import run_flow_eval

        def mock_get_all_paths():
            return [["Q1?"]]

        def mock_test_flow(questions, path_id, use_judge=True):
            return FlowResult(
                path_id=path_id,
                questions=questions,
                steps=[
                    FlowStepResult(
                        question="Q1?",
                        answer="Bad answer",
                        latency_ms=100,
                        has_answer=True,
                        relevance_score=0.5,
    
                    ),
                ],
                total_latency_ms=100,
                success=False,
                error="Failed test",
            )

        import backend.eval.integration.runner

        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1)

        assert results.paths_failed == 1
        assert len([r for r in results.all_results if not r.success]) == 1

    def test_run_flow_eval_with_answer_correctness(self, monkeypatch):
        """Test run_flow_eval with answer correctness metrics."""
        from backend.eval.integration.runner import run_flow_eval

        def mock_get_all_paths():
            return [["Q1?"]]

        def mock_test_flow(questions, path_id, use_judge=True):
            return FlowResult(
                path_id=path_id,
                questions=questions,
                steps=[
                    FlowStepResult(
                        question="Q1?",
                        answer="A1",
                        latency_ms=100,
                        has_answer=True,
                        relevance_score=0.9,
            
                        answer_correctness_score=0.85,
                    ),
                ],
                total_latency_ms=100,
                success=True,
            )

        import backend.eval.integration.runner

        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1)

        assert results.avg_answer_correctness == 0.85

    def test_run_flow_eval_multiple_paths(self, monkeypatch):
        """Test run_flow_eval with multiple paths."""
        from backend.eval.integration.runner import run_flow_eval

        def mock_get_all_paths():
            return [["Q1?"], ["Q2?"], ["Q3?"]]

        call_count = {"count": 0}

        def mock_test_flow(questions, path_id, use_judge=True):
            call_count["count"] += 1
            return FlowResult(
                path_id=path_id,
                questions=questions,
                steps=[
                    FlowStepResult(
                        question=questions[0],
                        answer=f"A{path_id}",
                        latency_ms=100,
                        has_answer=True,
                        relevance_score=0.9,
            
                    ),
                ],
                total_latency_ms=100,
                success=True,
            )

        import backend.eval.integration.runner

        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=3)

        assert results.paths_tested == 3
        assert call_count["count"] == 3


# =============================================================================
# LangSmith Edge Cases
# =============================================================================


class TestLangSmithEdgeCases:
    """Edge case tests for LangSmith module."""

    def test_get_latency_percentages_zero_total(self, monkeypatch):
        """Test get_latency_percentages when total avg is zero."""
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
                # Return agent nodes with zero latency
                return [
                    MockRun("route", now, now),  # 0ms
                    MockRun("answer", now, now),  # 0ms
                ]

        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)

        from backend.eval.integration.langsmith import get_latency_percentages

        result = get_latency_percentages()
        assert result == {}  # Zero total means empty result


# =============================================================================
# Tree Module Validation and Print Tests
# =============================================================================


class TestTreeValidation:
    """Tests for tree validation functions."""

    def test_get_tree_stats(self):
        """Test get_tree_stats returns expected keys."""
        from backend.eval.integration.tree import get_tree_stats

        stats = get_tree_stats()
        assert "num_starters" in stats
        assert "num_questions" in stats
        assert "num_edges" in stats
        assert "num_paths" in stats
        assert "max_depth" in stats
        assert "path_lengths" in stats

    def test_get_all_paths(self):
        """Test get_all_paths returns list of paths."""
        from backend.eval.integration.tree import get_all_paths

        paths = get_all_paths()
        assert isinstance(paths, list)
        if paths:
            assert isinstance(paths[0], list)


# =============================================================================
# Tree Module YAML Loading Tests
# =============================================================================


class TestYamlLoading:
    """Tests for YAML fixture loading."""

    def test_get_expected_answer_exists(self):
        """Test get_expected_answer for existing question."""
        from backend.eval.integration.tree import get_expected_answer

        # May return None if no fixtures, but shouldn't crash
        result = get_expected_answer("What deals are in the pipeline?")
        assert result is None or isinstance(result, str)

    def test_get_expected_answer_not_exists(self):
        """Test get_expected_answer for non-existing question."""
        from backend.eval.integration.tree import get_expected_answer

        result = get_expected_answer("This question does not exist in fixtures")
        assert result is None



# =============================================================================
# Judge Module Extended Tests
# =============================================================================


class TestJudgeModule:
    """Extended tests for judge module."""

    def test_suppress_event_loop_closed_errors(self):
        """Test install_event_loop_error_suppression doesn't crash."""
        from backend.eval.answer.text.suppression import install_event_loop_error_suppression

        # Should not raise
        install_event_loop_error_suppression()


# =============================================================================
# Tree Module YAML Error Handling Tests
# =============================================================================


class TestYamlLoadingErrors:
    """Tests for YAML loading error handling."""

    def test_load_yaml_fixture_nonexistent_file(self, monkeypatch, tmp_path):
        """Test _load_yaml_fixture with nonexistent file."""
        import backend.eval.integration.tree

        # Temporarily change fixtures path to a nonexistent directory
        monkeypatch.setattr(backend.eval.integration.tree, "_EVAL_FIXTURES_PATH", tmp_path / "nonexistent")

        # Should return empty dict without crashing
        result = backend.eval.integration.tree._load_yaml_fixture("test.yaml")
        assert result == {}

    def test_load_yaml_fixture_invalid_yaml(self, monkeypatch, tmp_path):
        """Test _load_yaml_fixture with invalid YAML content."""
        import backend.eval.integration.tree

        # Create a file with invalid YAML
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("::invalid:: yaml: [content")

        monkeypatch.setattr(backend.eval.integration.tree, "_EVAL_FIXTURES_PATH", tmp_path)

        # Should return empty dict without crashing
        result = backend.eval.integration.tree._load_yaml_fixture("invalid.yaml")
        assert result == {}

    def test_load_yaml_fixture_empty_file(self, monkeypatch, tmp_path):
        """Test _load_yaml_fixture with empty file."""
        import backend.eval.integration.tree

        # Create an empty file
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")

        monkeypatch.setattr(backend.eval.integration.tree, "_EVAL_FIXTURES_PATH", tmp_path)

        # Should return empty dict
        result = backend.eval.integration.tree._load_yaml_fixture("empty.yaml")
        assert result == {}


# =============================================================================
# Tree Module Path Finding Edge Cases
# =============================================================================


class TestTreePathFinding:
    """Tests for tree path finding edge cases."""

    def test_compute_max_depth_no_descendants(self, monkeypatch):
        """Test _compute_max_depth when starters have no descendants."""
        import networkx as nx

        import backend.eval.integration.tree

        # Create a minimal graph with only starters (no descendants)
        mock_g = nx.DiGraph()
        mock_g.add_node("Starter Q?")
        monkeypatch.setattr(backend.eval.integration.tree, "_G", mock_g)
        monkeypatch.setattr(backend.eval.integration.tree, "_STARTERS", ["Starter Q?"])

        result = backend.eval.integration.tree._compute_max_depth(["Starter Q?"])
        assert result == 0

    def test_find_paths_with_nx_no_path(self, monkeypatch):
        """Test _find_paths handles NetworkXNoPath exception."""
        import networkx as nx

        import backend.eval.integration.tree

        # Create a simple graph with a path
        mock_g = nx.DiGraph()
        mock_g.add_edge("Starter?", "Child?")
        mock_g.add_edge("Child?", "Leaf?")

        monkeypatch.setattr(backend.eval.integration.tree, "_G", mock_g)

        # Create subgraph with all nodes
        subgraph = mock_g.subgraph(["Starter?", "Child?", "Leaf?"])

        # This should work - leaf is Leaf?
        result = backend.eval.integration.tree._find_paths(["Starter?"], subgraph, 5)
        assert len(result) > 0  # Should find paths


# =============================================================================
# CLI Module Extended Tests
# =============================================================================


class TestCliModuleExtended:
    """Extended tests for CLI module."""

    def test_fetch_command(self, monkeypatch):
        """Test fetch command runs without error."""
        from backend.eval.fetch.runner import main as fetch_main

        def mock_run_sql_eval(**kwargs):
            from dataclasses import dataclass

            @dataclass
            class MockResults:
                total: int = 10
                passed: int = 9
                failed: int = 1
                pass_rate: float = 0.9
                cases: list = None

                def __post_init__(self):
                    self.cases = []

            return MockResults()

        def mock_print_summary(results):
            pass

        import backend.eval.fetch.runner
        monkeypatch.setattr(backend.eval.fetch.runner, "run_sql_eval", mock_run_sql_eval)
        monkeypatch.setattr(backend.eval.fetch.runner, "print_summary", mock_print_summary)

        # Should not raise
        fetch_main(limit=1, verbose=False)

    def test_main_command(self, monkeypatch):
        """Test main command calls _run_eval."""
        from backend.eval.integration.runner import main

        call_args = {}

        def mock_run_eval(**kwargs):
            call_args.update(kwargs)

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_run_eval", mock_run_eval)

        main(limit=5)

        assert call_args["limit"] == 5


# =============================================================================
# Tree Network Path Edge Cases
# =============================================================================


class TestTreeNetworkPaths:
    """Test tree path finding with network exceptions."""

    def test_compute_max_depth_no_path_between_nodes(self, monkeypatch):
        """Test _compute_max_depth when no path exists between starter and descendant."""
        import networkx as nx

        import backend.eval.integration.tree as tree_module

        # Create a graph with disconnected components
        G = nx.DiGraph()
        G.add_edge("starter", "node1")
        G.add_node("orphan")  # Node with no connection

        # Mock the global _G
        monkeypatch.setattr(tree_module, "_G", G)
        monkeypatch.setattr(tree_module, "_STARTERS", ["starter"])

        # This should handle the case where descendants are found but no path exists
        max_depth = tree_module._compute_max_depth(["starter"])
        assert max_depth >= 1  # At least depth 1 for starter -> node1

    def test_find_paths_disconnected_nodes(self, monkeypatch):
        """Test _find_paths with disconnected subgraph."""
        import networkx as nx

        import backend.eval.integration.tree as tree_module

        # Create a graph where path finding might fail
        G = nx.DiGraph()
        G.add_edge("starter", "mid")
        G.add_edge("mid", "leaf")

        # Mock the global _G
        monkeypatch.setattr(tree_module, "_G", G)
        monkeypatch.setattr(tree_module, "_STARTERS", ["starter"])

        # Test with disconnected leaf (not in the subgraph)
        subgraph = G.subgraph(["starter", "mid", "leaf"]).copy()
        paths = tree_module._find_paths(["starter"], subgraph, 3)
        assert len(paths) >= 1


# =============================================================================
# Runner Module Extended Coverage Tests
# =============================================================================


class TestRunnerEdgeCases:
    """Extended tests for runner edge cases."""

    def test_test_single_question_with_sql_results(self, monkeypatch):
        """Test test_single_question with SQL results."""
        from backend.eval.integration.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Test answer with good content",
                "sql_results": {"query": [{"company": "Acme", "value": 100}]},
            })

        def mock_evaluate_single(*args, **kwargs):
            return {
                "answer_relevancy": 0.90,
                "answer_correctness": 0.75,
                "nan_metrics": [],
            }

        def mock_get_expected_answer(question):
            return "Expected answer"

        import backend.eval.integration.runner

        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Test Q?", "session1")

        assert result.relevance_score == 0.90

    def test_test_single_question_ragas_failed(self, monkeypatch):
        """Test test_single_question with RAGAS failure."""
        from backend.eval.integration.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Test answer with good content",
                "sql_results": {"data": [{"id": 1}]},
            })

        def mock_evaluate_single(*args, **kwargs):
            return {
                "answer_relevancy": 0.0,
                "answer_correctness": 0.0,
                "nan_metrics": ["answer_relevancy", "answer_correctness"],
            }

        def mock_get_expected_answer(question):
            return "Expected"

        import backend.eval.integration.runner

        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Q?", "session1")

        assert result.ragas_metrics_failed >= 2

    def test_test_single_question_no_expected_answer(self, monkeypatch):
        """Test test_single_question without expected answer."""
        from backend.eval.integration.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Test answer with content",
                "sql_results": {"query": [{"a": 1}]},
            })

        def mock_evaluate_single(*args, **kwargs):
            return {
                "answer_relevancy": 0.85,
                "answer_correctness": 0.0,
                "nan_metrics": [],
            }

        def mock_get_expected_answer(question):
            return None

        import backend.eval.integration.runner

        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "evaluate_single", mock_evaluate_single)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Q?", "session1")

        assert result.relevance_score == 0.85


# =============================================================================
# Additional Coverage Tests for ragas.py
# =============================================================================


class TestRagasSuppression:
    """Tests for RAGAS error suppression in ragas.py."""

    def test_suppress_event_loop_closed_errors_already_run(self):
        """Test that install_event_loop_error_suppression can be called multiple times."""
        from backend.eval.answer.text.suppression import install_event_loop_error_suppression

        # The function should already have been called at import time
        # Calling it again should not raise
        install_event_loop_error_suppression()

    def test_event_loop_closed_filter(self):
        """Test EventLoopClosedFilter filters correctly."""
        import logging

        from backend.eval.answer.text import ragas

        # Create a mock log record
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Event loop is closed",
            args=(),
            exc_info=None,
        )

        # The filter should be registered - verify asyncio logger has it
        asyncio_logger = logging.getLogger("asyncio")
        filters = asyncio_logger.filters

        # There should be at least one filter that catches event loop errors
        assert any(hasattr(f, "filter") for f in filters)

    def test_ragas_executor_filter(self):
        """Test RagasExecutorFilter filters correctly."""
        import logging

        # The ragas.executor logger should have the filter
        executor_logger = logging.getLogger("ragas.executor")
        filters = executor_logger.filters

        # Should have at least one filter
        assert any(hasattr(f, "filter") for f in filters)


class TestRagasEvaluateSingle:
    """Tests for ragas.py evaluate_single function edge cases."""

    @pytest.mark.no_mock_llm
    def test_evaluate_single_empty_contexts(self, monkeypatch):
        """Test evaluate_single with empty contexts."""
        from unittest.mock import MagicMock, patch

        import pandas as pd

        from backend.eval.answer.text import ragas

        # Create a DataFrame to return (only answer_correctness now)
        df = pd.DataFrame({
            "answer_correctness": [0.75],
            "answer_relevancy": [0.80],
        })

        mock_eval_result = MagicMock()
        mock_eval_result.to_pandas.return_value = df

        with patch.object(ragas, "evaluate", return_value=mock_eval_result):
            with patch.object(ragas, "_evaluators", return_value=(MagicMock(), MagicMock())):
                result = ragas.evaluate_single(
                    question="What is the capital?",
                    answer="Paris",
                    contexts=[],  # Empty contexts
                    reference_answer="Paris is the capital of France",
                )

        # Should return answer_correctness score
        assert "answer_correctness" in result

    @pytest.mark.no_mock_llm
    def test_evaluate_single_with_nan_metrics(self, monkeypatch):
        """Test evaluate_single tracks NaN metrics."""
        from unittest.mock import MagicMock, patch

        import pandas as pd

        from backend.eval.answer.text import ragas

        # Create a DataFrame with NaN value for answer_correctness
        df = pd.DataFrame({
            "answer_correctness": [float("nan")],
            "answer_relevancy": [float("nan")],
        })

        mock_eval_result = MagicMock()
        mock_eval_result.to_pandas.return_value = df

        with patch.object(ragas, "evaluate", return_value=mock_eval_result):
            with patch.object(ragas, "_evaluators", return_value=(MagicMock(), MagicMock())):
                result = ragas.evaluate_single(
                    question="Test?",
                    answer="Answer",
                    contexts=["Context"],
                    reference_answer="Expected answer",
                )

        # Should track nan_metrics
        assert "answer_correctness" in result["nan_metrics"]
        assert result["answer_correctness"] == 0.0

    @pytest.mark.no_mock_llm
    def test_evaluators_returns_two_metrics(self, monkeypatch):
        """Test _evaluators returns 2 metrics (AnswerCorrectness + AnswerRelevancy)."""
        from unittest.mock import MagicMock, patch

        from backend.eval.answer.text import ragas

        # Mock the LLM at module level
        with patch.object(ragas, "get_langchain_chat_openai", return_value=MagicMock()):
            # Clear cache and get fresh evaluators
            ragas._evaluators.cache_clear()
            metrics = ragas._evaluators()

        # Should have 2 metrics: AnswerCorrectness + AnswerRelevancy
        assert len(metrics) == 2

        # Clean up cache
        ragas._evaluators.cache_clear()

    @pytest.mark.no_mock_llm
    def test_extract_scores_with_nan_values(self, monkeypatch):
        """Test _extract_scores with NaN values."""
        from unittest.mock import MagicMock

        import pandas as pd

        from backend.eval.answer.text import ragas

        # Create a DataFrame with NaN value
        df = pd.DataFrame({
            "answer_correctness": [None],
            "answer_relevancy": [None],
        })

        # Create mock eval_result that returns the DataFrame
        mock_eval_result = MagicMock()
        mock_eval_result.to_pandas.return_value = df

        result = ragas._extract_scores(mock_eval_result)

        # None value should be converted to 0.0
        assert result["answer_correctness"] == 0.0
        # Should track nan_metrics
        assert "answer_correctness" in result["nan_metrics"]


class TestMainMiddleware:
    """Tests for main.py middleware."""

    @pytest.mark.asyncio
    @pytest.mark.no_mock_llm
    async def test_request_logging_middleware(self):
        """Test RequestLoggingMiddleware logs requests."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi import FastAPI, Request, Response
        from starlette.testclient import TestClient

        from backend.main import RequestLoggingMiddleware

        # Create a simple app with middleware
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
