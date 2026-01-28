"""
Tests for backend.eval module.

Tests the evaluation models and shared utilities.
"""

import os
from typing import Any

import pytest

# Set mock mode
os.environ["MOCK_LLM"] = "1"


def make_invoke_mock(result: dict[str, Any]) -> dict[str, Any]:
    """Helper to create _invoke_agent return value."""
    return result


from backend.eval.integration.models import (
    SLO_CONVO_STEP_PASS_RATE,
    ConvoEvalResults,
    ConvoStepResult,
)

# =============================================================================
# ConvoStepResult Model Tests
# =============================================================================


class TestConvoStepResult:
    """Tests for ConvoStepResult."""

    def test_creation(self):
        result = ConvoStepResult(
            question="What is Acme's revenue?",
            answer="Acme's revenue is $1M.",
            relevance_score=0.9,
        )
        assert result.relevance_score == 0.9

    def test_passed_property(self):
        passing = ConvoStepResult(
            question="Q", answer="A",
            relevance_score=0.8, answer_correctness_score=0.8, ragas_metrics_total=2,
        )
        assert passing.passed is True

        failing_relevance = ConvoStepResult(
            question="Q", answer="A",
            relevance_score=0.5, answer_correctness_score=0.8, ragas_metrics_total=2,
        )
        assert failing_relevance.passed is False

        failing_correctness = ConvoStepResult(
            question="Q", answer="A",
            relevance_score=0.8, answer_correctness_score=0.5, ragas_metrics_total=2,
        )
        assert failing_correctness.passed is False

        passing_no_ragas = ConvoStepResult(
            question="Q", answer="A",
            relevance_score=0.0, ragas_metrics_total=0,
        )
        assert passing_no_ragas.passed is True

    def test_passed_with_errors(self):
        result = ConvoStepResult(
            question="Q", answer="A",
            relevance_score=0.9, errors=["Something went wrong"],
        )
        assert result.passed is False

    def test_passed_with_action_failed(self):
        result = ConvoStepResult(
            question="Q", answer="A",
            action_passed=False,
        )
        assert result.passed is False

    def test_action_missing(self):
        result = ConvoStepResult(
            question="Q", answer="A",
            expected_action=True, suggested_action=None,
        )
        assert result.action_missing is True
        assert result.action_spurious is False

    def test_action_spurious(self):
        result = ConvoStepResult(
            question="Q", answer="A",
            expected_action=False, suggested_action="Do something",
        )
        assert result.action_spurious is True
        assert result.action_missing is False

    def test_action_no_expectation(self):
        result = ConvoStepResult(
            question="Q", answer="A",
            expected_action=None, suggested_action="Do something",
        )
        assert result.action_missing is False
        assert result.action_spurious is False


# =============================================================================
# ConvoEvalResults Model Tests
# =============================================================================


class TestConvoEvalResults:
    """Tests for ConvoEvalResults (extends BaseEvalResults)."""

    def test_creation(self):
        results = ConvoEvalResults(
            total=10, passed=8,
            avg_relevance=0.85,
        )
        assert results.passed == 8
        assert results.avg_relevance == 0.85

    def test_pass_rate(self):
        results = ConvoEvalResults(total=10, passed=8)
        assert results.pass_rate == 0.8

    def test_failed_property(self):
        results = ConvoEvalResults(total=10, passed=8)
        assert results.failed == 2

    def test_empty(self):
        results = ConvoEvalResults(total=0, passed=0)
        assert results.pass_rate == 0.0

    def test_compute_aggregates(self):
        cases = [
            ConvoStepResult(
                question="Q1?", answer="A1",
                relevance_score=0.9, answer_correctness_score=0.8,
                ragas_metrics_total=2, ragas_metrics_failed=0,
            ),
            ConvoStepResult(
                question="Q2?", answer="A2",
                relevance_score=0.5, answer_correctness_score=0.3,
                ragas_metrics_total=2, ragas_metrics_failed=1,
            ),
        ]
        results = ConvoEvalResults(total=2, cases=cases)
        results.compute_aggregates()

        assert results.passed == 1
        assert results.failed == 1
        assert results.avg_relevance == 0.7
        assert results.avg_answer_correctness == 0.55
        assert results.ragas_metrics_total == 4
        assert results.ragas_metrics_failed == 1

    def test_compute_aggregates_with_actions(self):
        cases = [
            ConvoStepResult(
                question="Q1?", answer="A1",
                suggested_action="Schedule call", action_passed=True,
                action_relevance=0.9, action_actionability=0.8, action_appropriateness=0.85,
            ),
            ConvoStepResult(
                question="Q2?", answer="A2",
                suggested_action="Follow up", action_passed=False,
                action_relevance=0.5, action_actionability=0.4, action_appropriateness=0.3,
            ),
            ConvoStepResult(
                question="Q3?", answer="A3",
                expected_action=True, suggested_action=None, action_passed=False,
            ),
            ConvoStepResult(
                question="Q4?", answer="A4",
                expected_action=False, suggested_action="Spurious", action_passed=False,
            ),
        ]
        results = ConvoEvalResults(total=4, cases=cases)
        results.compute_aggregates()

        assert results.actions_judged == 3  # Q1, Q2, Q4 have suggested_action
        assert results.actions_passed == 1  # Q1
        assert results.actions_missing == 1  # Q3
        assert results.actions_spurious == 1  # Q4
        assert results.avg_action_relevance == pytest.approx((0.9 + 0.5 + 0.0) / 3)


# =============================================================================
# SLO Constants Tests
# =============================================================================


class TestSLOConstants:
    """Tests for SLO constant values."""

    def test_slo_convo_step_pass_rate(self):
        assert SLO_CONVO_STEP_PASS_RATE == 0.95


# =============================================================================
# RAGAS Success Rate Tests
# =============================================================================


class TestModelsExtended:
    """Extended tests for models module."""

    def test_ragas_success_rate_zero_total(self):
        results = ConvoEvalResults(total=1, passed=1, ragas_metrics_total=0, ragas_metrics_failed=0)
        assert results.ragas_success_rate == 1.0

    def test_ragas_success_rate_all_failed(self):
        results = ConvoEvalResults(total=1, passed=0, ragas_metrics_total=5, ragas_metrics_failed=5)
        assert results.ragas_success_rate == 0.0

    def test_ragas_success_rate_partial_success(self):
        results = ConvoEvalResults(total=1, passed=1, ragas_metrics_total=10, ragas_metrics_failed=3)
        assert results.ragas_success_rate == 0.7


# =============================================================================
# Output Module Tests
# =============================================================================


class TestOutputModule:
    """Tests for print_summary function."""

    def test_print_summary_all_pass(self):
        from backend.eval.integration.runner import print_summary
        results = ConvoEvalResults(
            total=10, passed=9,
            avg_relevance=0.90, avg_answer_correctness=0.75,
            ragas_metrics_total=100, ragas_metrics_failed=5,
        )
        result = print_summary(results)
        assert isinstance(result, bool)

    def test_print_summary_no_actions(self):
        from backend.eval.integration.runner import print_summary
        results = ConvoEvalResults(total=5, passed=4, avg_relevance=0.90)
        result = print_summary(results)
        assert isinstance(result, bool)

    def test_print_summary_with_failed_questions(self):
        from backend.eval.integration.runner import print_summary
        cases = [
            ConvoStepResult(
                question="Q1?", answer="",
                errors=["Agent crashed"],
            ),
        ]
        results = ConvoEvalResults(total=1, cases=cases)
        results.compute_aggregates()
        result = print_summary(results)
        assert result is False

    def test_print_summary_with_actions(self):
        from backend.eval.integration.runner import print_summary
        results = ConvoEvalResults(
            total=5, passed=5,
            actions_judged=3, actions_passed=2,
            actions_missing=1, actions_spurious=0,
            avg_action_relevance=0.85, avg_action_actionability=0.80,
            avg_action_appropriateness=0.90,
        )
        result = print_summary(results)
        assert result is True


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
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_action", lambda q: None)

        result = test_single_question("What is Acme's status?", "session1")
        assert result.relevance_score == 0.90

    def test_no_answer(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({"answer": "", "sql_results": {}})

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: None)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_action", lambda q: None)

        result = test_single_question("Hello?", "session1", use_judge=False)
        assert result.relevance_score == 0.0

    def test_exception(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            raise RuntimeError("Agent crashed")

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)

        result = test_single_question("Q?", "session1")
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
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_action", lambda q: None)

        result = test_single_question("Q?", "session1")
        assert result.relevance_score == 0.85

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
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_action", lambda q: None)

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
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_action", lambda q: None)

        result = test_single_question("Q?", "session1")
        assert result.relevance_score == 0.0  # RAGAS failed, defaults to 0

    def test_action_judged(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Good answer with content",
                "sql_results": {},
                "suggested_actions": ["Schedule a call"],
            })
        def mock_judge(*args):
            return (True, 0.9, 0.8, 0.85, "Good action")

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: None)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_action", lambda q: True)
        monkeypatch.setattr(backend.eval.integration.runner, "judge_suggested_action", mock_judge)

        result = test_single_question("Q?", "session1")
        assert result.suggested_action == "Schedule a call"
        assert result.action_passed is True
        assert result.action_relevance == 0.9

    def test_action_missing(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Answer without action",
                "sql_results": {},
                "suggested_actions": [],
            })

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: None)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_action", lambda q: True)

        result = test_single_question("Q?", "session1", use_judge=False)
        assert result.action_passed is False
        assert result.action_missing is True

    def test_action_spurious(self, monkeypatch):
        from backend.eval.integration.runner import test_single_question
        def mock_invoke_agent(question, session_id=None):
            return make_invoke_mock({
                "answer": "Answer with unwanted action",
                "sql_results": {},
                "suggested_actions": ["Unwanted action"],
            })

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_answer", lambda q: None)
        monkeypatch.setattr(backend.eval.integration.runner, "get_expected_action", lambda q: False)

        result = test_single_question("Q?", "session1", use_judge=False)
        assert result.action_passed is False
        assert result.action_spurious is True


# =============================================================================
# Runner Module Tests (run_convo_eval)
# =============================================================================


class TestRunConvoEval:
    """Tests for run_convo_eval function."""

    def test_basic(self, monkeypatch):
        from backend.eval.integration.runner import run_convo_eval
        def mock_get_all_paths(): return [["Q1?", "Q2?"]]
        call_count = {"count": 0}
        def mock_test_single_question(question, session_id, use_judge=True):
            call_count["count"] += 1
            return ConvoStepResult(
                question=question, answer=f"Answer to {question}",
                relevance_score=0.9,
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_single_question", mock_test_single_question)

        results = run_convo_eval(max_paths=1)
        assert results.total == 2
        assert results.passed == 2
        assert call_count["count"] == 2

    def test_with_failures(self, monkeypatch):
        from backend.eval.integration.runner import run_convo_eval
        def mock_get_all_paths(): return [["Q1?"]]
        def mock_test_single_question(question, session_id, use_judge=True):
            return ConvoStepResult(
                question=question, answer="Bad",
                relevance_score=0.5, answer_correctness_score=0.3,
                ragas_metrics_total=2,
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_single_question", mock_test_single_question)

        results = run_convo_eval(max_paths=1)
        assert results.failed == 1

    def test_with_answer_correctness(self, monkeypatch):
        from backend.eval.integration.runner import run_convo_eval
        def mock_get_all_paths(): return [["Q1?"]]
        def mock_test_single_question(question, session_id, use_judge=True):
            return ConvoStepResult(
                question=question, answer="A1",
                relevance_score=0.9, answer_correctness_score=0.85,
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_single_question", mock_test_single_question)

        results = run_convo_eval(max_paths=1)
        assert results.avg_answer_correctness == 0.85

    def test_multiple_paths(self, monkeypatch):
        from backend.eval.integration.runner import run_convo_eval
        def mock_get_all_paths(): return [["Q1?"], ["Q2?"], ["Q3?"]]
        call_count = {"count": 0}
        def mock_test_single_question(question, session_id, use_judge=True):
            call_count["count"] += 1
            return ConvoStepResult(
                question=question, answer=f"A",
                relevance_score=0.9,
            )

        import backend.eval.integration.runner
        monkeypatch.setattr(backend.eval.integration.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.integration.runner, "test_single_question", mock_test_single_question)

        results = run_convo_eval(max_paths=3)
        assert results.total == 3
        assert call_count["count"] == 3


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

    def test_get_expected_action_exists(self):
        from backend.eval.integration.tree import get_expected_action
        result = get_expected_action("What deals are in the pipeline?")
        assert result is True

    def test_get_expected_action_false(self):
        from backend.eval.integration.tree import get_expected_action
        result = get_expected_action("How are deals distributed by stage?")
        assert result is False

    def test_get_expected_action_not_exists(self):
        from backend.eval.integration.tree import get_expected_action
        assert get_expected_action("This question does not exist") is None


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
