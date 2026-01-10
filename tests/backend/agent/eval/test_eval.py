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
            sql_success_rate=0.95,
            sql_query_count=100,
            rag_decision_accuracy=0.90,
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

    def test_ensure_qdrant_collections_ingest_failure(self, monkeypatch, tmp_path):
        """Test ensure_qdrant_collections when ingest fails to create collection."""
        state = {"ingested": False}

        class MockCollection:
            points_count = 0

        class MockClient:
            def collection_exists(self, name):
                return False  # Always returns False (ingest failed)

            def get_collection(self, name):
                return MockCollection()

        def mock_get_client():
            return MockClient()

        def mock_close_client():
            pass

        def mock_ingest_private():
            state["ingested"] = True  # Ingest runs but collection not created

        import backend.agent.rag.client
        import backend.agent.rag.ingest
        import backend.agent.rag.config

        monkeypatch.setattr(backend.agent.rag.client, "get_qdrant_client", mock_get_client)
        monkeypatch.setattr(backend.agent.rag.client, "close_qdrant_client", mock_close_client)
        monkeypatch.setattr(backend.agent.rag.ingest, "ingest_private_texts", mock_ingest_private)
        monkeypatch.setattr(backend.agent.rag.config, "QDRANT_PATH", tmp_path / "qdrant")

        from backend.eval.cli import ensure_qdrant_collections

        with pytest.raises(RuntimeError, match="Failed to create collection"):
            ensure_qdrant_collections()


# =============================================================================
# Output Module Tests
# =============================================================================


class TestOutputModule:
    """Tests for backend.eval.output module."""

    def test_print_summary_all_pass(self):
        """Test print_summary with all SLOs passing."""
        from backend.eval.output import print_summary

        results = FlowEvalResults(
            total_paths=10,
            paths_tested=10,
            paths_passed=9,
            paths_failed=1,
            total_questions=30,
            questions_passed=28,
            questions_failed=2,
            sql_success_rate=0.98,
            sql_query_count=50,
            rag_decision_accuracy=0.92,
            avg_relevance=0.90,
            avg_faithfulness=0.92,
            avg_answer_correctness=0.75,
            avg_account_precision=0.85,
            avg_account_recall=0.75,
            account_sample_count=15,
            rag_invoked_count=20,
            avg_latency_per_question_ms=3000,
            latency_routing_pct=0.20,
            latency_retrieval_pct=0.30,
            latency_answer_pct=0.25,
            ragas_metrics_total=100,
            ragas_metrics_failed=5,
        )

        # Should not raise and should return a boolean
        result = print_summary(results, eval_mode="both")
        assert isinstance(result, bool)

    def test_print_summary_rag_mode(self):
        """Test print_summary with eval_mode='rag'."""
        from backend.eval.output import print_summary

        results = FlowEvalResults(
            total_paths=5,
            paths_tested=5,
            paths_passed=4,
            paths_failed=1,
            total_questions=15,
            questions_passed=13,
            questions_failed=2,
            avg_account_precision=0.85,
            avg_account_recall=0.80,
            account_sample_count=10,
            rag_invoked_count=12,
        )

        result = print_summary(results, eval_mode="rag")
        assert isinstance(result, bool)

    def test_print_summary_pipeline_mode(self):
        """Test print_summary with eval_mode='pipeline'."""
        from backend.eval.output import print_summary

        results = FlowEvalResults(
            total_paths=5,
            paths_tested=5,
            paths_passed=4,
            paths_failed=1,
            total_questions=15,
            questions_passed=13,
            questions_failed=2,
            avg_relevance=0.90,
            avg_faithfulness=0.88,
            avg_answer_correctness=0.72,
        )

        result = print_summary(results, eval_mode="pipeline")
        assert isinstance(result, bool)

    def test_print_summary_no_sql_queries(self):
        """Test print_summary when no SQL queries executed."""
        from backend.eval.output import print_summary

        results = FlowEvalResults(
            total_paths=5,
            paths_tested=5,
            paths_passed=4,
            paths_failed=1,
            total_questions=15,
            questions_passed=13,
            questions_failed=2,
            sql_query_count=0,  # No SQL queries executed
            rag_decision_accuracy=0.95,
            avg_relevance=0.90,
            avg_faithfulness=0.92,
        )

        result = print_summary(results, eval_mode="both")
        assert isinstance(result, bool)

    def test_count_ragas_failures_pipeline_mode(self):
        """Test _count_ragas_failures in pipeline mode."""
        from backend.eval.output import _count_ragas_failures

        step = FlowStepResult(
            question="Q",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.5,  # Below SLO
            faithfulness_score=0.5,  # Below SLO
            answer_correctness_score=0.5,  # Below SLO
        )

        count = _count_ragas_failures(step, eval_mode="pipeline")
        assert count == 3  # All 3 metrics failed

    def test_count_ragas_failures_rag_mode(self):
        """Test _count_ragas_failures in rag mode."""
        from backend.eval.output import _count_ragas_failures

        step = FlowStepResult(
            question="Q",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            account_rag_invoked=True,
            account_precision_score=0.5,  # Below SLO
            account_recall_score=0.5,  # Below SLO
        )

        count = _count_ragas_failures(step, eval_mode="rag")
        assert count == 2  # Both RAG metrics failed

    def test_count_ragas_failures_both_mode(self):
        """Test _count_ragas_failures in both mode."""
        from backend.eval.output import _count_ragas_failures

        step = FlowStepResult(
            question="Q",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.5,  # Below SLO
            faithfulness_score=0.95,  # Passing
            answer_correctness_score=0.5,  # Below SLO
            account_rag_invoked=True,
            account_precision_score=0.5,  # Below SLO
            account_recall_score=0.85,  # Passing
        )

        count = _count_ragas_failures(step, eval_mode="both")
        assert count == 3  # relevance, answer_correctness, precision failed

    def test_count_ragas_failures_rag_not_invoked(self):
        """Test _count_ragas_failures when RAG not invoked."""
        from backend.eval.output import _count_ragas_failures

        step = FlowStepResult(
            question="Q",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            account_rag_invoked=False,  # RAG not invoked
            account_precision_score=0.0,
            account_recall_score=0.0,
        )

        count = _count_ragas_failures(step, eval_mode="rag")
        assert count == 0  # RAG not invoked, so no failures counted

    def test_save_results(self, tmp_path):
        """Test save_results writes correct JSON."""
        from backend.eval.output import save_results

        results = FlowEvalResults(
            total_paths=5,
            paths_tested=5,
            paths_passed=4,
            paths_failed=1,
            total_questions=15,
            questions_passed=13,
            questions_failed=2,
            sql_success_rate=0.95,
            sql_query_count=30,
            rag_decision_accuracy=0.85,
            avg_relevance=0.88,
            avg_faithfulness=0.92,
            avg_answer_correctness=0.70,
            avg_account_precision=0.80,
            avg_account_recall=0.75,
            total_latency_ms=5000,
            avg_latency_per_question_ms=333.3,
            wall_clock_ms=10000,
            latency_routing_pct=0.20,
            latency_retrieval_pct=0.30,
            latency_answer_pct=0.25,
            ragas_metrics_total=50,
            ragas_metrics_failed=3,
        )

        output_path = tmp_path / "results.json"
        save_results(results, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert "summary" in data
        assert "slo_results" in data
        assert data["summary"]["composite_score"] == results.composite_score

    def test_save_results_with_failed_paths(self, tmp_path):
        """Test save_results includes failed path details."""
        from backend.eval.output import save_results

        failed_step = FlowStepResult(
            question="Failed question?",
            answer="Bad answer",
            latency_ms=1000,
            has_answer=True,
            has_sources=False,
            relevance_score=0.4,
            faithfulness_score=0.5,
            answer_correctness_score=0.3,
            account_precision_score=0.2,
            account_recall_score=0.1,
            judge_explanation="Poor quality answer",
            error=None,
        )

        failed_flow = FlowResult(
            path_id=1,
            questions=["Failed question?"],
            steps=[failed_step],
            total_latency_ms=1000,
            success=False,
        )

        results = FlowEvalResults(
            total_paths=2,
            paths_tested=2,
            paths_passed=1,
            paths_failed=1,
            total_questions=2,
            questions_passed=1,
            questions_failed=1,
            failed_paths=[failed_flow],
        )

        output_path = tmp_path / "results_with_failures.json"
        save_results(results, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["failed_paths"]) == 1
        assert data["failed_paths"][0]["path_id"] == 1
        assert data["failed_paths"][0]["steps"][0]["question"] == "Failed question?"

    def test_check_qdrant_access_success(self, monkeypatch):
        """Test check_qdrant_access returns True when accessible."""
        from backend.eval.output import check_qdrant_access

        class MockClient:
            def get_collections(self):
                return []

        def mock_get_client():
            return MockClient()

        import backend.agent.rag.client
        monkeypatch.setattr(backend.agent.rag.client, "get_qdrant_client", mock_get_client)

        result = check_qdrant_access()
        assert result is True

    def test_check_qdrant_access_locked(self, monkeypatch):
        """Test check_qdrant_access returns False when locked."""
        from backend.eval.output import check_qdrant_access

        def mock_get_client():
            raise Exception("Database already accessed by another process")

        import backend.agent.rag.client
        monkeypatch.setattr(backend.agent.rag.client, "get_qdrant_client", mock_get_client)

        result = check_qdrant_access()
        assert result is False

    def test_check_qdrant_access_other_error(self, monkeypatch):
        """Test check_qdrant_access returns True for non-lock errors."""
        from backend.eval.output import check_qdrant_access

        def mock_get_client():
            raise Exception("Connection timeout")

        import backend.agent.rag.client
        monkeypatch.setattr(backend.agent.rag.client, "get_qdrant_client", mock_get_client)

        result = check_qdrant_access()
        assert result is True  # Non-lock errors are treated as accessible


class TestPrintSloFailures:
    """Tests for _print_slo_failures function."""

    def test_print_slo_failures_no_failures(self):
        """Test _print_slo_failures with no failures."""
        from backend.eval.output import _print_slo_failures

        results = FlowEvalResults(
            total_paths=1,
            paths_tested=1,
            paths_passed=1,
            paths_failed=0,
            total_questions=1,
            questions_passed=1,
            questions_failed=0,
            all_results=[
                FlowResult(
                    path_id=0,
                    questions=["Q1"],
                    steps=[
                        FlowStepResult(
                            question="Q1",
                            answer="A1",
                            latency_ms=100,
                            has_answer=True,
                            has_sources=True,
                            relevance_score=0.95,
                            faithfulness_score=0.95,
                            answer_correctness_score=0.80,
                        )
                    ],
                    total_latency_ms=100,
                    success=True,
                )
            ],
        )

        # Should not raise - just returns early
        _print_slo_failures(results, eval_mode="both")

    def test_print_slo_failures_with_failures(self):
        """Test _print_slo_failures with failures."""
        from backend.eval.output import _print_slo_failures

        results = FlowEvalResults(
            total_paths=1,
            paths_tested=1,
            paths_passed=0,
            paths_failed=1,
            total_questions=2,
            questions_passed=0,
            questions_failed=2,
            all_results=[
                FlowResult(
                    path_id=0,
                    questions=["Q1", "Q2"],
                    steps=[
                        FlowStepResult(
                            question="Q1 is a very long question that should be truncated",
                            answer="A1",
                            latency_ms=100,
                            has_answer=True,
                            has_sources=True,
                            relevance_score=0.5,  # Below SLO
                            faithfulness_score=0.5,  # Below SLO
                            answer_correctness_score=0.3,
                        ),
                        FlowStepResult(
                            question="Q2",
                            answer="A2",
                            latency_ms=100,
                            has_answer=True,
                            has_sources=True,
                            relevance_score=0.4,
                            faithfulness_score=0.4,
                            answer_correctness_score=0.2,
                            account_rag_invoked=True,
                            account_precision_score=0.3,
                            account_recall_score=0.2,
                        ),
                    ],
                    total_latency_ms=200,
                    success=False,
                )
            ],
        )

        # Should not raise
        _print_slo_failures(results, eval_mode="both")
        _print_slo_failures(results, eval_mode="rag")
        _print_slo_failures(results, eval_mode="pipeline")


# =============================================================================
# Runner Module Tests
# =============================================================================


class TestRunnerModule:
    """Tests for backend.eval.runner module."""

    def test_judge_answer_success_with_mock(self):
        """Test judge_answer uses mock mode and returns valid scores."""
        from backend.eval.runner import judge_answer

        # In mock mode, judge_answer uses _mock_evaluate_single
        # A long answer with context should return good scores
        result = judge_answer(
            "What is the status?",
            "The status is active and working correctly.",
            ["Status data: Active"]
        )
        # Mock mode returns: relevance=0.85, faithfulness=0.80
        assert result["relevance"] == 0.85
        assert result["faithfulness"] == 0.80
        assert result["ragas_failed"] is False

    def test_judge_answer_without_context(self):
        """Test judge_answer with empty context returns reduced scores."""
        from backend.eval.runner import judge_answer

        result = judge_answer(
            "What is the status?",
            "The status is unknown at this time.",
            []  # Empty context
        )
        # Mock mode returns: relevance=0.70, faithfulness=0.50 without context
        assert result["relevance"] == 0.70
        assert result["faithfulness"] == 0.50
        assert result["ragas_failed"] is False

    def test_judge_answer_short_answer(self):
        """Test judge_answer with short answer returns zeros."""
        from backend.eval.runner import judge_answer

        result = judge_answer(
            "What?",
            "Yes",  # Too short (< 10 chars)
            ["Context"]
        )
        # Mock mode returns zeros for short answers
        assert result["relevance"] == 0.0
        assert result["faithfulness"] == 0.0
        assert result["ragas_failed"] is False  # No error, just low scores

    def test_judge_answer_with_reference(self):
        """Test judge_answer with reference answer."""
        from backend.eval.runner import judge_answer

        result = judge_answer(
            "What is the revenue?",
            "The revenue is approximately $1M for this quarter.",
            ["Revenue data shows $1M"],
            reference_answer="The Q4 revenue was $1 million.",
        )
        # Mock mode includes context_recall and answer_correctness when reference provided
        assert result["context_recall"] == 0.70
        assert result["answer_correctness"] == 0.65


class TestTestSingleQuestion:
    """Tests for test_single_question function."""

    def test_test_single_question_success(self, monkeypatch):
        """Test test_single_question with successful execution."""
        from backend.eval.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return {
                "answer": "This is a good answer with enough content.",
                "sources": [{"type": "note", "id": "1"}],
                "resolved_company_id": "ACME001",
                "needs_account_rag": True,
                "account_chunks": ["context chunk"],
                "account_rag_invoked": True,
                "sql_results": {"company_info": [{"name": "Acme", "company_id": "ACME001"}]},
                "sql_queries_total": 2,
                "sql_queries_success": 2,
                "account_context_answer": "Account context from RAG",
            }

        def mock_get_expected_answer(question):
            return "Expected answer"

        def mock_judge_answer(*args, **kwargs):
            return {
                "relevance": 0.90,
                "faithfulness": 0.85,
                "context_precision": 0.80,
                "context_recall": 0.75,
                "answer_correctness": 0.70,
                "explanation": "",
                "ragas_failed": False,
                "nan_metrics": [],
            }

        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.runner, "judge_answer", mock_judge_answer)

        import backend.agent.followup.tree
        monkeypatch.setattr(backend.agent.followup.tree, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("What is Acme's status?", [], "session1")

        assert result.has_answer is True
        assert result.sql_queries_total == 2
        assert result.sql_queries_success == 2
        assert result.relevance_score == 0.90

    def test_test_single_question_no_answer(self, monkeypatch):
        """Test test_single_question when agent returns no answer."""
        from backend.eval.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return {
                "answer": "",  # Empty answer
                "sources": [],
                "intent": "general",
                "sql_queries_total": 0,
                "sql_queries_success": 0,
            }

        def mock_get_expected_answer(question):
            return None

        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "_invoke_agent", mock_invoke_agent)

        import backend.agent.followup.tree
        monkeypatch.setattr(backend.agent.followup.tree, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Hello?", [], "session1", use_judge=False)

        assert result.has_answer is False
        assert result.relevance_score == 0.0

    def test_test_single_question_exception(self, monkeypatch):
        """Test test_single_question handles exceptions."""
        from backend.eval.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            raise RuntimeError("Agent crashed")

        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "_invoke_agent", mock_invoke_agent)

        result = test_single_question("Q?", [], "session1")

        assert result.has_answer is False
        assert result.error is not None
        assert "Agent crashed" in result.error

    def test_test_single_question_rag_mode(self, monkeypatch):
        """Test test_single_question in rag eval mode."""
        from backend.eval.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return {
                "answer": "Good answer with sufficient length.",
                "sources": [],
                "intent": "account_context",
                "account_chunks": ["chunk1", "chunk2"],
                "account_rag_invoked": True,
                "account_context_answer": "context",
                "sql_queries_total": 1,
                "sql_queries_success": 1,
            }

        def mock_get_expected_answer(question):
            return "Expected"

        def mock_judge_answer(*args, **kwargs):
            return {
                "relevance": 0.0,
                "faithfulness": 0.0,
                "context_precision": 0.85,
                "context_recall": 0.80,
                "answer_correctness": 0.0,
                "explanation": "",
                "ragas_failed": False,
                "nan_metrics": [],
            }

        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.runner, "judge_answer", mock_judge_answer)

        import backend.agent.followup.tree
        monkeypatch.setattr(backend.agent.followup.tree, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Q?", [], "session1", eval_mode="rag")

        assert result.account_rag_invoked is True
        assert result.account_precision_score == 0.85

    def test_test_single_question_pipeline_mode(self, monkeypatch):
        """Test test_single_question in pipeline eval mode."""
        from backend.eval.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return {
                "answer": "Good answer with sufficient length for testing.",
                "sources": [],
                "needs_account_rag": False,
                "sql_results": {
                    "pipeline": [{"stage": "Discovery", "count": 5}],
                },
                "sql_queries_total": 1,
                "sql_queries_success": 1,
            }

        def mock_get_expected_answer(question):
            return "Expected"

        def mock_judge_answer(*args, **kwargs):
            return {
                "relevance": 0.90,
                "faithfulness": 0.88,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.75,
                "explanation": "",
                "ragas_failed": False,
                "nan_metrics": [],
            }

        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.runner, "judge_answer", mock_judge_answer)

        import backend.agent.followup.tree
        monkeypatch.setattr(backend.agent.followup.tree, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Pipeline?", [], "session1", eval_mode="pipeline")

        assert result.relevance_score == 0.90
        assert result.faithfulness_score == 0.88


class TestTestFlow:
    """Tests for test_flow function."""

    def test_test_flow_success(self, monkeypatch):
        """Test test_flow with successful execution."""
        from backend.eval.runner import test_flow

        call_count = {"count": 0}

        def mock_test_single_question(question, history, session_id, use_judge=True, verbose=False, eval_mode="both"):
            call_count["count"] += 1
            return FlowStepResult(
                question=question,
                answer=f"Answer to {question}",
                latency_ms=100,
                has_answer=True,
                has_sources=True,
                relevance_score=0.90,
                faithfulness_score=0.85,
            )

        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "test_single_question", mock_test_single_question)

        result = test_flow(["Q1?", "Q2?"], path_id=0)

        assert result.success is True
        assert len(result.steps) == 2
        assert result.total_latency_ms == 200
        assert call_count["count"] == 2

    def test_test_flow_with_failure(self, monkeypatch):
        """Test test_flow when a step fails."""
        from backend.eval.runner import test_flow

        def mock_test_single_question(question, history, session_id, use_judge=True, verbose=False, eval_mode="both"):
            if "fail" in question.lower():
                return FlowStepResult(
                    question=question,
                    answer="",
                    latency_ms=100,
                    has_answer=False,
                    has_sources=False,
                    relevance_score=0.0,
                    faithfulness_score=0.0,
                    error="Failed to answer",
                )
            return FlowStepResult(
                question=question,
                answer="Good answer",
                latency_ms=100,
                has_answer=True,
                has_sources=True,
                relevance_score=0.90,
                faithfulness_score=0.85,
            )

        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "test_single_question", mock_test_single_question)

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
                    MockRun("route", now, now + timedelta(milliseconds=100)),
                    MockRun("fetch_account", now, now + timedelta(milliseconds=400)),
                    MockRun("answer", now, now + timedelta(milliseconds=300)),
                    MockRun("followup", now, now + timedelta(milliseconds=200)),
                ]

        import sys
        mock_langsmith = type(sys)("langsmith")
        mock_langsmith.Client = MockClient
        monkeypatch.setitem(sys.modules, "langsmith", mock_langsmith)

        from backend.eval.langsmith import get_latency_percentages

        result = get_latency_percentages()

        assert "routing" in result
        assert "retrieval" in result
        assert "answer" in result
        # Total = 100 + 400 + 300 + 200 = 1000
        assert abs(result["routing"] - 0.10) < 0.01  # 100/1000
        assert abs(result["retrieval"] - 0.40) < 0.01  # 400/1000

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

        from backend.eval.langsmith import get_latency_percentages

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

        from backend.eval.langsmith import get_latency_percentages

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
        from backend.eval.langsmith import get_latency_breakdown
        result = get_latency_breakdown()
        assert result == {}


# =============================================================================
# Formatting Tests (Extended)
# =============================================================================


class TestFormattingExtended:
    """Extended tests for formatting module."""

    def test_build_eval_table_with_aggregate_row(self):
        """Test build_eval_table with aggregate row."""
        from backend.eval.formatting import build_eval_table

        sections = [
            (
                "Quality",
                [
                    ("Relevance", "85%", ">=80%", True),
                ],
            ),
        ]

        table = build_eval_table(
            "Test Table",
            sections,
            aggregate_row=("Total", "90%", ">=85%", True),
        )
        assert table.title == "Test Table"

    def test_build_eval_table_with_failing_aggregate(self):
        """Test build_eval_table with failing aggregate row."""
        from backend.eval.formatting import build_eval_table

        sections = [
            (
                "Quality",
                [
                    ("Relevance", "50%", ">=80%", False),
                ],
            ),
        ]

        table = build_eval_table(
            "Test Table",
            sections,
            aggregate_row=("Total", "50%", ">=85%", False),
        )
        assert table.title == "Test Table"

    def test_build_eval_table_empty_section_name(self):
        """Test build_eval_table with empty section name."""
        from backend.eval.formatting import build_eval_table

        sections = [
            (
                "",  # Empty section name
                [
                    ("Metric1", "90%", ">=80%", True),
                ],
            ),
        ]

        table = build_eval_table("Test", sections)
        assert table.title == "Test"

    def test_build_eval_table_no_slo_target(self):
        """Test build_eval_table with None slo_target."""
        from backend.eval.formatting import build_eval_table

        sections = [
            (
                "Metrics",
                [
                    ("Tracked Only", "50%", None, None),  # No SLO target
                ],
            ),
        ]

        table = build_eval_table("Test", sections)
        assert table.title == "Test"

    def test_print_debug_failures_with_format_item(self):
        """Test print_debug_failures with custom format_item function."""
        from backend.eval.formatting import print_debug_failures

        format_called = {"count": 0}

        def custom_format(i, item):
            format_called["count"] += 1

        failures = [
            {"id": "t1", "error": "Error 1"},
            {"id": "t2", "error": "Error 2"},
        ]

        print_debug_failures(failures, "Test", format_item=custom_format)
        assert format_called["count"] == 2


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

    def test_flow_step_result_low_faithfulness(self):
        """Test FlowStepResult.passed with low faithfulness."""
        result = FlowStepResult(
            question="Q",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.9,  # High
            faithfulness_score=0.5,  # Below 0.7 threshold
        )

        assert result.passed is False


# =============================================================================
# Judge Module Tests (Extended)
# =============================================================================


class TestJudgeExtended:
    """Extended tests for judge module."""

    def test_mock_evaluate_single_no_reference(self):
        """Test mock evaluate without reference answer."""
        from backend.eval.judge import _mock_evaluate_single

        result = _mock_evaluate_single(
            question="What is the status?",
            answer="The status is active with enough content.",
            contexts=["Status is active"],
            reference_answer=None,  # No reference
        )

        assert result["context_recall"] == 0.0  # No reference = no recall
        assert result["answer_correctness"] == 0.0  # No reference = no correctness

    def test_mock_evaluate_single_short_answer(self):
        """Test mock evaluate with short answer (< 10 chars)."""
        from backend.eval.judge import _mock_evaluate_single

        result = _mock_evaluate_single(
            question="What?",
            answer="Yes",  # Too short (< 10 chars)
            contexts=["Some context"],
        )

        assert result["answer_relevancy"] == 0.0  # Short answer = no relevancy

    def test_mock_evaluate_single_no_context_placeholder(self):
        """Test mock evaluate with 'No context provided' placeholder."""
        from backend.eval.judge import _mock_evaluate_single

        result = _mock_evaluate_single(
            question="What?",
            answer="This is a good answer with enough content.",
            contexts=["No context provided"],  # Placeholder context
        )

        # Should be treated as no context
        assert result["context_precision"] == 0.0
        assert result["answer_relevancy"] == 0.70  # Reduced score


# =============================================================================
# CLI Module Tests (Extended)
# =============================================================================


class TestCliModule:
    """Tests for backend.eval.cli module."""

    def test_run_eval_qdrant_not_accessible(self, monkeypatch):
        """Test _run_eval when Qdrant is not accessible."""
        from backend.eval.cli import _run_eval

        def mock_check_qdrant_access():
            return False

        import backend.eval.cli
        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)

        # Should return early (not raise) when Qdrant not accessible
        _run_eval(
            limit=1,
            verbose=False,
            no_judge=True,
            output=None,
            debug=False,
            eval_mode="both",
        )

    def test_run_eval_warmup_failure(self, monkeypatch, tmp_path):
        """Test _run_eval handles warmup failure gracefully."""
        from backend.eval.cli import _run_eval

        def mock_check_qdrant_access():
            return True

        def mock_tool_entity_rag(*args, **kwargs):
            raise RuntimeError("Warmup failed")

        def mock_get_tree_stats():
            return {"total_paths": 10}

        def mock_run_flow_eval(**kwargs):
            return FlowEvalResults(
                total_paths=1,
                paths_tested=1,
                paths_passed=1,
                paths_failed=0,
                total_questions=1,
                questions_passed=1,
                questions_failed=0,
            )

        def mock_get_latency_percentages(**kwargs):
            return {}

        import backend.eval.cli
        import backend.agent.rag.tools
        import backend.agent.followup.tree
        import backend.eval.runner
        import backend.eval.langsmith

        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.agent.rag.tools, "tool_entity_rag", mock_tool_entity_rag)
        monkeypatch.setattr(backend.agent.followup.tree, "get_tree_stats", mock_get_tree_stats)
        monkeypatch.setattr(backend.eval.runner, "run_flow_eval", mock_run_flow_eval)
        monkeypatch.setattr(backend.eval.langsmith, "get_latency_percentages", mock_get_latency_percentages)

        # Should complete without raising
        _run_eval(
            limit=1,
            verbose=False,
            no_judge=True,
            output=None,
            debug=False,
            eval_mode="both",
        )

    def test_run_eval_with_output_file(self, monkeypatch, tmp_path):
        """Test _run_eval saves output file."""
        from backend.eval.cli import _run_eval

        def mock_check_qdrant_access():
            return True

        def mock_tool_entity_rag(*args, **kwargs):
            return "", []

        def mock_get_tree_stats():
            return {"total_paths": 10}

        def mock_run_flow_eval(**kwargs):
            return FlowEvalResults(
                total_paths=1,
                paths_tested=1,
                paths_passed=1,
                paths_failed=0,
                total_questions=1,
                questions_passed=1,
                questions_failed=0,
            )

        def mock_get_latency_percentages(**kwargs):
            return {"routing": 0.2, "retrieval": 0.3, "answer": 0.25}

        import backend.eval.cli
        import backend.agent.rag.tools
        import backend.agent.followup.tree
        import backend.eval.runner
        import backend.eval.langsmith

        # Must patch in cli module where it's imported
        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.agent.rag.tools, "tool_entity_rag", mock_tool_entity_rag)
        monkeypatch.setattr(backend.agent.followup.tree, "get_tree_stats", mock_get_tree_stats)
        monkeypatch.setattr(backend.eval.runner, "run_flow_eval", mock_run_flow_eval)
        monkeypatch.setattr(backend.eval.langsmith, "get_latency_percentages", mock_get_latency_percentages)

        output_path = tmp_path / "results.json"

        _run_eval(
            limit=1,
            verbose=False,
            no_judge=True,
            output=str(output_path),
            debug=False,
            eval_mode="both",
        )

        assert output_path.exists()

    def test_run_eval_with_debug_failures(self, monkeypatch, tmp_path):
        """Test _run_eval with debug mode and failed paths."""
        from backend.eval.cli import _run_eval

        def mock_check_qdrant_access():
            return True

        def mock_tool_entity_rag(*args, **kwargs):
            return "", []

        def mock_get_tree_stats():
            return {"total_paths": 10}

        failed_step = FlowStepResult(
            question="Failed Q?",
            answer="Bad A",
            latency_ms=100,
            has_answer=True,
            has_sources=False,
            relevance_score=0.5,
            faithfulness_score=0.5,
            judge_explanation="Low quality",
        )
        failed_flow = FlowResult(
            path_id=1,
            questions=["Failed Q?"],
            steps=[failed_step],
            total_latency_ms=100,
            success=False,
        )

        def mock_run_flow_eval(**kwargs):
            return FlowEvalResults(
                total_paths=1,
                paths_tested=1,
                paths_passed=0,
                paths_failed=1,
                total_questions=1,
                questions_passed=0,
                questions_failed=1,
                failed_paths=[failed_flow],
            )

        def mock_get_latency_percentages(**kwargs):
            return {}

        import backend.eval.cli
        import backend.agent.rag.tools
        import backend.agent.followup.tree
        import backend.eval.runner
        import backend.eval.langsmith

        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.agent.rag.tools, "tool_entity_rag", mock_tool_entity_rag)
        monkeypatch.setattr(backend.agent.followup.tree, "get_tree_stats", mock_get_tree_stats)
        monkeypatch.setattr(backend.eval.runner, "run_flow_eval", mock_run_flow_eval)
        monkeypatch.setattr(backend.eval.langsmith, "get_latency_percentages", mock_get_latency_percentages)

        # Should print debug failures
        _run_eval(
            limit=1,
            verbose=False,
            no_judge=True,
            output=None,
            debug=True,  # Enable debug output
            eval_mode="both",
        )

    def test_run_eval_exception_handling(self, monkeypatch):
        """Test _run_eval handles exceptions gracefully."""
        from backend.eval.cli import _run_eval

        def mock_check_qdrant_access():
            return True

        def mock_tool_entity_rag(*args, **kwargs):
            return "", []

        def mock_get_tree_stats():
            return {"total_paths": 10}

        def mock_run_flow_eval(**kwargs):
            raise RuntimeError("Evaluation crashed")

        import backend.eval.cli
        import backend.agent.rag.tools
        import backend.agent.followup.tree
        import backend.eval.runner

        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.agent.rag.tools, "tool_entity_rag", mock_tool_entity_rag)
        monkeypatch.setattr(backend.agent.followup.tree, "get_tree_stats", mock_get_tree_stats)
        monkeypatch.setattr(backend.eval.runner, "run_flow_eval", mock_run_flow_eval)

        # Should handle exception gracefully (not raise)
        _run_eval(
            limit=1,
            verbose=False,
            no_judge=True,
            output=None,
            debug=False,
            eval_mode="both",
        )


# =============================================================================
# Runner Module Tests (Extended - run_flow_eval)
# =============================================================================


class TestRunFlowEval:
    """Tests for run_flow_eval function."""

    def test_run_flow_eval_basic(self, monkeypatch):
        """Test run_flow_eval with basic execution."""
        from backend.eval.runner import run_flow_eval

        def mock_get_paths_for_role(role=None):
            return [["Q1?", "Q2?"]]

        def mock_test_flow(questions, path_id, use_judge=True, verbose=False, eval_mode="both"):
            return FlowResult(
                path_id=path_id,
                questions=questions,
                steps=[
                    FlowStepResult(
                        question="Q1?",
                        answer="A1",
                        latency_ms=100,
                        has_answer=True,
                        has_sources=True,
                        relevance_score=0.9,
                        faithfulness_score=0.9,
                        sql_queries_total=2,
                        sql_queries_success=2,
                        rag_decision_correct=True,
                    ),
                    FlowStepResult(
                        question="Q2?",
                        answer="A2",
                        latency_ms=100,
                        has_answer=True,
                        has_sources=True,
                        relevance_score=0.85,
                        faithfulness_score=0.85,
                        sql_queries_total=1,
                        sql_queries_success=1,
                    ),
                ],
                total_latency_ms=200,
                success=True,
            )

        import backend.eval.runner

        monkeypatch.setattr(backend.eval.runner, "get_paths_for_role", mock_get_paths_for_role)
        monkeypatch.setattr(backend.eval.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1, concurrency=1)

        assert results.total_paths == 1
        assert results.paths_passed == 1
        assert results.total_questions == 2

    def test_run_flow_eval_with_failures(self, monkeypatch):
        """Test run_flow_eval with failed paths."""
        from backend.eval.runner import run_flow_eval

        def mock_get_paths_for_role(role=None):
            return [["Q1?"]]

        def mock_test_flow(questions, path_id, use_judge=True, verbose=False, eval_mode="both"):
            return FlowResult(
                path_id=path_id,
                questions=questions,
                steps=[
                    FlowStepResult(
                        question="Q1?",
                        answer="Bad answer",
                        latency_ms=100,
                        has_answer=True,
                        has_sources=False,
                        relevance_score=0.5,
                        faithfulness_score=0.5,
                    ),
                ],
                total_latency_ms=100,
                success=False,
                error="Failed test",
            )

        import backend.eval.runner

        monkeypatch.setattr(backend.eval.runner, "get_paths_for_role", mock_get_paths_for_role)
        monkeypatch.setattr(backend.eval.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1, concurrency=1)

        assert results.paths_failed == 1
        assert len(results.failed_paths) == 1

    def test_run_flow_eval_with_rag_metrics(self, monkeypatch):
        """Test run_flow_eval with RAG metrics."""
        from backend.eval.runner import run_flow_eval

        def mock_get_paths_for_role(role=None):
            return [["Q1?"]]

        def mock_test_flow(questions, path_id, use_judge=True, verbose=False, eval_mode="both"):
            return FlowResult(
                path_id=path_id,
                questions=questions,
                steps=[
                    FlowStepResult(
                        question="Q1?",
                        answer="A1",
                        latency_ms=100,
                        has_answer=True,
                        has_sources=True,
                        relevance_score=0.9,
                        faithfulness_score=0.9,
                        account_rag_invoked=True,
                        account_precision_score=0.85,
                        account_recall_score=0.80,
                        precision_succeeded=True,
                        recall_succeeded=True,
                    ),
                ],
                total_latency_ms=100,
                success=True,
            )

        import backend.eval.runner

        monkeypatch.setattr(backend.eval.runner, "get_paths_for_role", mock_get_paths_for_role)
        monkeypatch.setattr(backend.eval.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1, concurrency=1, eval_mode="rag")

        assert results.avg_account_precision == 0.85
        assert results.avg_account_recall == 0.80
        assert results.account_sample_count == 1

    def test_run_flow_eval_parallel(self, monkeypatch):
        """Test run_flow_eval with parallel execution."""
        from backend.eval.runner import run_flow_eval

        def mock_get_paths_for_role(role=None):
            return [["Q1?"], ["Q2?"], ["Q3?"]]

        call_count = {"count": 0}

        def mock_test_flow(questions, path_id, use_judge=True, verbose=False, eval_mode="both"):
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
                        has_sources=True,
                        relevance_score=0.9,
                        faithfulness_score=0.9,
                    ),
                ],
                total_latency_ms=100,
                success=True,
            )

        import backend.eval.runner

        monkeypatch.setattr(backend.eval.runner, "get_paths_for_role", mock_get_paths_for_role)
        monkeypatch.setattr(backend.eval.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=3, concurrency=2)

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

        from backend.eval.langsmith import get_latency_percentages

        result = get_latency_percentages()
        assert result == {}  # Zero total means empty result
