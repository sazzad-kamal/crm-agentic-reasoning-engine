"""
Tests for backend.eval module.

Tests the evaluation models, formatting, and shared utilities.
"""

import json
import os

import pytest

# Set mock mode
os.environ["MOCK_LLM"] = "1"

from backend.eval.models import (
    _latency_score,
    FlowStepResult,
    FlowResult,
    FlowEvalResults,
    SLO_ROUTER_ACCURACY,
    SLO_FLOW_PATH_PASS_RATE,
    SLO_FLOW_QUESTION_PASS_RATE,
    SLO_FLOW_RELEVANCE,
    SLO_FLOW_FAITHFULNESS,
)


# =============================================================================
# Latency Score Helper Tests
# =============================================================================


class TestLatencyScore:
    """Tests for _latency_score helper function."""

    def test_latency_score_at_slo(self):
        """Test latency score at exactly SLO target."""
        assert _latency_score(3000.0, 3000.0) == 1.0

    def test_latency_score_below_slo(self):
        """Test latency score below SLO target."""
        assert _latency_score(2000.0, 3000.0) == 1.0

    def test_latency_score_at_double_slo(self):
        """Test latency score at 2x SLO target."""
        assert _latency_score(6000.0, 3000.0) == 0.0

    def test_latency_score_above_double_slo(self):
        """Test latency score above 2x SLO target."""
        assert _latency_score(9000.0, 3000.0) == 0.0

    def test_latency_score_interpolation(self):
        """Test latency score linear interpolation between SLO and 2x SLO."""
        # 4500ms is halfway between 3000ms (SLO) and 6000ms (2x SLO)
        # Score should be 0.5
        assert _latency_score(4500.0, 3000.0) == 0.5

    def test_latency_score_interpolation_quarter(self):
        """Test latency score at 25% above SLO."""
        # 3750ms is 25% of the way from 3000ms to 6000ms
        # Score should be 0.75
        assert _latency_score(3750.0, 3000.0) == 0.75


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
            has_sources=True,
            relevance_score=0.9,
            faithfulness_score=0.85,
        )

        assert result.relevance_score == 0.9
        assert result.faithfulness_score == 0.85

    def test_flow_step_result_passed_property(self):
        """Test FlowStepResult.passed property."""
        # Passing case
        passing = FlowStepResult(
            question="Q",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.8,
            faithfulness_score=0.8,
        )
        assert passing.passed is True

        # Failing - low relevance
        failing_relevance = FlowStepResult(
            question="Q",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.5,
            faithfulness_score=0.8,
        )
        assert failing_relevance.passed is False

        # Failing - no answer
        failing_no_answer = FlowStepResult(
            question="Q",
            answer="",
            latency_ms=100,
            has_answer=False,
            has_sources=False,
            relevance_score=0.8,
            faithfulness_score=0.8,
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
                has_sources=True,
                relevance_score=0.9,
                faithfulness_score=0.9,
            ),
            FlowStepResult(
                question="Q2",
                answer="A2",
                latency_ms=150,
                has_answer=True,
                has_sources=True,
                relevance_score=0.85,
                faithfulness_score=0.85,
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
            avg_faithfulness=0.80,
            total_latency_ms=5000,
            avg_latency_per_question_ms=166.7,
            p95_latency_ms=300.0,
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
        assert results.question_pass_rate == 0.9

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
        assert results.question_pass_rate == 0.0

    def test_flow_eval_results_composite_score(self):
        """Test FlowEvalResults composite score calculation."""
        results = FlowEvalResults(
            total_paths=10,
            paths_tested=10,
            paths_passed=9,
            paths_failed=1,
            total_questions=40,
            questions_passed=36,
            questions_failed=4,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.90,
            avg_relevance=0.88,
            avg_faithfulness=0.92,
            avg_answer_correctness=0.75,
            avg_account_precision=0.85,
            avg_account_recall=0.80,
            avg_latency_per_question_ms=3000.0,  # Under SLO, so latency_score = 1.0
        )

        # Composite = 0.30*faith + 0.20*rel + 0.15*ans + 0.10*acct_prec + 0.10*acct_recall + 0.10*routing + 0.05*latency
        # routing = (0.95 + 0.90) / 2 = 0.925
        # latency_score = 1.0 (3000ms <= 4000ms SLO)
        # = 0.30*0.92 + 0.20*0.88 + 0.15*0.75 + 0.10*0.85 + 0.10*0.80 + 0.10*0.925 + 0.05*1.0
        # = 0.276 + 0.176 + 0.1125 + 0.085 + 0.080 + 0.0925 + 0.05 = 0.872
        expected = 0.30 * 0.92 + 0.20 * 0.88 + 0.15 * 0.75 + 0.10 * 0.85 + 0.10 * 0.80 + 0.10 * 0.925 + 0.05 * 1.0
        assert abs(results.composite_score - expected) < 0.001
        assert results.composite_score > 0.85  # Should pass SLO


# =============================================================================
# SLO Constants Tests
# =============================================================================


class TestSLOConstants:
    """Tests for SLO constant values."""

    def test_slo_router_accuracy(self):
        """Test SLO router accuracy threshold."""
        assert SLO_ROUTER_ACCURACY == 0.90

    def test_slo_flow_path_pass_rate(self):
        """Test SLO flow path pass rate threshold."""
        assert SLO_FLOW_PATH_PASS_RATE == 0.85

    def test_slo_flow_question_pass_rate(self):
        """Test SLO flow question pass rate threshold."""
        assert SLO_FLOW_QUESTION_PASS_RATE == 0.90

    def test_slo_flow_relevance(self):
        """Test SLO flow relevance threshold."""
        assert SLO_FLOW_RELEVANCE == 0.85

    def test_slo_flow_faithfulness(self):
        """Test SLO flow faithfulness threshold."""
        assert SLO_FLOW_FAITHFULNESS == 0.90


# =============================================================================
# Formatting Tests
# =============================================================================


class TestFormatters:
    """Tests for formatting functions."""

    def test_format_check_mark_true(self):
        """Test format_check_mark with True."""
        from backend.eval.formatting import format_check_mark

        result = format_check_mark(True)
        assert "[green]Y[/green]" in result

    def test_format_check_mark_false(self):
        """Test format_check_mark with False."""
        from backend.eval.formatting import format_check_mark

        result = format_check_mark(False)
        assert "[red]X[/red]" in result

    def test_format_percentage_high(self):
        """Test format_percentage with high value (green)."""
        from backend.eval.formatting import format_percentage

        result = format_percentage(0.95)
        assert "[green]" in result
        assert "95.0%" in result

    def test_format_percentage_medium(self):
        """Test format_percentage with medium value (yellow)."""
        from backend.eval.formatting import format_percentage

        result = format_percentage(0.75)
        assert "[yellow]" in result
        assert "75.0%" in result

    def test_format_percentage_low(self):
        """Test format_percentage with low value (red)."""
        from backend.eval.formatting import format_percentage

        result = format_percentage(0.50)
        assert "[red]" in result
        assert "50.0%" in result

    def test_format_percentage_custom_thresholds(self):
        """Test format_percentage with custom thresholds."""
        from backend.eval.formatting import format_percentage

        result = format_percentage(0.75, thresholds=(0.8, 0.6))
        assert "[yellow]" in result


class TestTables:
    """Tests for table creation functions."""

    def test_create_summary_table(self):
        """Test create_summary_table creates valid table."""
        from backend.eval.formatting import create_summary_table

        table = create_summary_table("Test Summary")
        assert table.title == "Test Summary"
        assert len(table.columns) == 2

    def test_build_eval_table(self):
        """Test build_eval_table creates valid table."""
        from backend.eval.formatting import build_eval_table

        sections = [
            (
                "Quality",
                [
                    ("Relevance", "85%", ">=80%", True),
                    ("Faithfulness", "75%", ">=80%", False),
                ],
            ),
        ]

        table = build_eval_table("Test Table", sections)
        assert table.title == "Test Table"
        assert len(table.columns) == 3  # Metric, Value, SLO


class TestPrintFunctions:
    """Tests for print/output functions (verify they don't crash)."""

    def test_print_eval_header(self):
        """Test print_eval_header runs without error."""
        from backend.eval.formatting import print_eval_header

        print_eval_header("Test Header", "Test Subtitle")

    def test_print_overall_result_panel_pass(self):
        """Test print_overall_result_panel with pass."""
        from backend.eval.formatting import print_overall_result_panel

        print_overall_result_panel(True, [], "All tests passed!")

    def test_print_overall_result_panel_fail(self):
        """Test print_overall_result_panel with failure."""
        from backend.eval.formatting import print_overall_result_panel

        print_overall_result_panel(False, ["SLO failed", "Regression detected"], "")

    def test_print_debug_failures_empty(self):
        """Test print_debug_failures with empty list."""
        from backend.eval.formatting import print_debug_failures

        print_debug_failures([], "No Failures")

    def test_print_debug_failures_with_items(self):
        """Test print_debug_failures with items."""
        from backend.eval.formatting import print_debug_failures

        failures = [
            {"id": "t1", "error": "Test error 1"},
            {"id": "t2", "error": "Test error 2"},
        ]

        print_debug_failures(failures, "Test Failures")


# =============================================================================
# Parallel Runner Tests
# =============================================================================


class TestLatencyCalculation:
    """Tests for latency calculation functions."""

    def test_calculate_p95_latency_empty_list(self):
        """Test calculate_p95_latency with empty list."""
        from backend.eval.parallel import calculate_p95_latency

        assert calculate_p95_latency([]) == 0.0

    def test_calculate_p95_latency_single_value(self):
        """Test calculate_p95_latency with single value."""
        from backend.eval.parallel import calculate_p95_latency

        assert calculate_p95_latency([1000]) == 1000.0

    def test_calculate_p95_latency_multiple_values(self):
        """Test calculate_p95_latency with multiple values."""
        from backend.eval.parallel import calculate_p95_latency

        latencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        p95 = calculate_p95_latency(latencies)
        assert p95 == 1000.0

    def test_calculate_p95_latency_with_outliers(self):
        """Test calculate_p95_latency with outliers."""
        from backend.eval.parallel import calculate_p95_latency

        latencies = [100] * 95 + [10000] * 5
        p95 = calculate_p95_latency(latencies)
        assert p95 >= 100


# =============================================================================
# RAGAS Judge Tests (with mocks)
# =============================================================================


@pytest.mark.skipif(
    os.environ.get("MOCK_LLM", "0") == "1",
    reason="RAGAS imports not available in MOCK_LLM mode",
)
class TestRagasJudge:
    """Tests for RAGAS judge with mocked evaluate."""

    def test_evaluate_single_success(self, monkeypatch):
        """Test evaluate_single with successful RAGAS call."""
        import pandas as pd

        # Mock EvaluationResult with to_pandas()
        class MockResult:
            def to_pandas(self):
                return pd.DataFrame([{
                    "answer_relevancy": 0.85,
                    "faithfulness": 0.90,
                }])

        def mock_evaluate(dataset, metrics, **kwargs):
            return MockResult()

        monkeypatch.setattr("backend.eval.judge.evaluate", mock_evaluate)

        from backend.eval.judge import evaluate_single

        result = evaluate_single(
            question="What is the revenue?",
            answer="The revenue is $1M.",
            contexts=["Revenue data shows $1M for Q4."],
        )

        assert result["answer_relevancy"] == 0.85
        assert result["faithfulness"] == 0.90

    def test_evaluate_single_empty_contexts(self, monkeypatch):
        """Test evaluate_single with empty contexts."""
        import pandas as pd

        class MockResult:
            def to_pandas(self):
                return pd.DataFrame([{
                    "answer_relevancy": 0.5,
                    "faithfulness": 0.5,
                }])

        def mock_evaluate(dataset, metrics, **kwargs):
            # Verify contexts is not empty (should be ["No context provided"])
            assert dataset["retrieved_contexts"][0] == ["No context provided"]
            return MockResult()

        monkeypatch.setattr("backend.eval.judge.evaluate", mock_evaluate)

        from backend.eval.judge import evaluate_single

        result = evaluate_single(
            question="What is the revenue?",
            answer="I don't know.",
            contexts=[],
        )

        assert result["answer_relevancy"] == 0.5

    def test_evaluate_single_exception(self, monkeypatch):
        """Test evaluate_single handles exceptions gracefully."""
        def mock_evaluate(dataset, metrics, **kwargs):
            raise RuntimeError("RAGAS API error")

        monkeypatch.setattr("backend.eval.judge.evaluate", mock_evaluate)

        from backend.eval.judge import evaluate_single

        result = evaluate_single(
            question="What is the revenue?",
            answer="The revenue is $1M.",
            contexts=["Some context"],
        )

        # Should return zeros on error
        assert result["answer_relevancy"] == 0.0
        assert result["faithfulness"] == 0.0
        assert result["context_precision"] == 0.0


# =============================================================================
# RAGAS Mock Mode Tests
# =============================================================================


class TestRagasMockMode:
    """Tests for RAGAS mock mode evaluation."""

    def test_mock_evaluate_single_with_context(self, monkeypatch):
        """Test mock evaluate returns scores when answer and context present."""
        monkeypatch.setenv("MOCK_LLM", "1")

        # Force reimport with mock mode enabled
        from backend.eval.judge import _mock_evaluate_single

        result = _mock_evaluate_single(
            question="What is the revenue?",
            answer="The revenue is $1M for Q4.",
            contexts=["Revenue data shows $1M."],
            reference_answer="The Q4 revenue was $1 million.",
        )

        assert result["answer_relevancy"] == 0.85
        assert result["faithfulness"] == 0.80
        assert result["context_precision"] == 0.75
        assert result["context_recall"] == 0.70
        assert result["answer_correctness"] == 0.65

    def test_mock_evaluate_single_without_context(self, monkeypatch):
        """Test mock evaluate returns reduced scores without context."""
        monkeypatch.setenv("MOCK_LLM", "1")

        from backend.eval.judge import _mock_evaluate_single

        result = _mock_evaluate_single(
            question="What is the revenue?",
            answer="The revenue is $1M for Q4.",
            contexts=[],
        )

        assert result["answer_relevancy"] == 0.70
        assert result["faithfulness"] == 0.50
        assert result["context_precision"] == 0.0
        assert result["context_recall"] == 0.0

    def test_mock_evaluate_single_empty_answer(self, monkeypatch):
        """Test mock evaluate returns zeros for empty answer."""
        monkeypatch.setenv("MOCK_LLM", "1")

        from backend.eval.judge import _mock_evaluate_single

        result = _mock_evaluate_single(
            question="What is the revenue?",
            answer="",
            contexts=["Some context"],
        )

        assert result["answer_relevancy"] == 0.0
        assert result["faithfulness"] == 0.0


# =============================================================================
# LangSmith Latency Tests (with mocks)
# =============================================================================


class TestLangSmithLatency:
    """Tests for LangSmith latency breakdown with mocks."""

    def test_get_latency_breakdown_no_api_key(self, monkeypatch):
        """Test get_latency_breakdown without API key."""
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        from backend.eval.langsmith import get_latency_breakdown

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

        from backend.eval.langsmith import get_latency_breakdown

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
                        MockRun("fetch_account", now, now + timedelta(milliseconds=500)),
                        MockRun("answer", now, now + timedelta(milliseconds=300)),
                    ]
                else:
                    # Parent runs
                    return [MockRun("agent", now, now + timedelta(seconds=1))]

        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)

        from backend.eval.langsmith import get_latency_breakdown

        result = get_latency_breakdown()

        assert "route" in result
        assert "fetch_account" in result
        assert "answer" in result
        assert result["route"]["avg_ms"] == 100.0
        assert result["fetch_account"]["avg_ms"] == 500.0
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

        from backend.eval.langsmith import get_latency_breakdown

        result = get_latency_breakdown()
        assert result == {}


# =============================================================================
# Ensure Qdrant Collections Tests (with mocks)
# =============================================================================


class TestEnsureQdrantCollections:
    """Tests for ensure_qdrant_collections with mocks."""

    def test_ensure_qdrant_collections_exists(self, monkeypatch, tmp_path):
        """Test ensure_qdrant_collections when collections exist."""

        class MockCollection:
            points_count = 10

        class MockClient:
            def collection_exists(self, name):
                return True

            def get_collection(self, name):
                return MockCollection()

        def mock_get_client():
            return MockClient()

        import backend.agent.rag.client
        import backend.agent.rag.config

        monkeypatch.setattr(backend.agent.rag.client, "get_qdrant_client", mock_get_client)
        monkeypatch.setattr(backend.agent.rag.config, "QDRANT_PATH", tmp_path / "qdrant")

        from backend.eval.cli import ensure_qdrant_collections

        # Should complete without calling ingest
        ensure_qdrant_collections()

    def test_ensure_qdrant_collections_missing(self, monkeypatch, tmp_path):
        """Test ensure_qdrant_collections when collections are missing."""
        # Track state: before ingest, collection doesn't exist; after ingest, it does
        state = {"ingested": False}

        class MockCollection:
            @property
            def points_count(self):
                return 102 if state["ingested"] else 0

        class MockClient:
            def collection_exists(self, name):
                return state["ingested"]

            def get_collection(self, name):
                return MockCollection()

        def mock_get_client():
            return MockClient()

        def mock_close_client():
            pass

        def mock_ingest_private():
            state["ingested"] = True

        import backend.agent.rag.client
        import backend.agent.rag.ingest
        import backend.agent.rag.config

        monkeypatch.setattr(backend.agent.rag.client, "get_qdrant_client", mock_get_client)
        monkeypatch.setattr(backend.agent.rag.client, "close_qdrant_client", mock_close_client)
        monkeypatch.setattr(backend.agent.rag.ingest, "ingest_private_texts", mock_ingest_private)
        monkeypatch.setattr(backend.agent.rag.config, "QDRANT_PATH", tmp_path / "qdrant")

        from backend.eval.cli import ensure_qdrant_collections

        # Should call ingest functions and verify collection was created
        ensure_qdrant_collections()
        assert state["ingested"]
