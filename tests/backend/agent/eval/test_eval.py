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
            rag_detection_rate=0.90,
            avg_relevance=0.88,
            avg_faithfulness=0.92,
            avg_answer_correctness=0.75,
            avg_account_precision=0.85,
            avg_account_recall=0.80,
            avg_latency_per_question_ms=3000.0,  # Under SLO, so latency_score = 1.0
        )

        # Composite = 0.30*faith + 0.20*rel + 0.15*ans + 0.10*acct_prec + 0.10*acct_recall + 0.10*sql + 0.05*latency
        # latency_score = 1.0 (3000ms <= 4000ms SLO)
        # = 0.30*0.92 + 0.20*0.88 + 0.15*0.75 + 0.10*0.85 + 0.10*0.80 + 0.10*0.95 + 0.05*1.0
        expected = 0.30 * 0.92 + 0.20 * 0.88 + 0.15 * 0.75 + 0.10 * 0.85 + 0.10 * 0.80 + 0.10 * 0.95 + 0.05 * 1.0
        assert abs(results.composite_score - expected) < 0.001
        assert results.composite_score > 0.85  # Should pass SLO


# =============================================================================
# SLO Constants Tests
# =============================================================================


class TestSLOConstants:
    """Tests for SLO constant values."""

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

        from backend.eval.langsmith import get_latency_breakdown

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

        import backend.agent.fetch.rag.client
        import backend.agent.fetch.rag.config

        monkeypatch.setattr(backend.agent.fetch.rag.client, "get_qdrant_client", mock_get_client)
        monkeypatch.setattr(backend.agent.fetch.rag.config, "QDRANT_PATH", tmp_path / "qdrant")

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

        import backend.agent.fetch.rag.client
        import backend.agent.fetch.rag.ingest
        import backend.agent.fetch.rag.config

        monkeypatch.setattr(backend.agent.fetch.rag.client, "get_qdrant_client", mock_get_client)
        monkeypatch.setattr(backend.agent.fetch.rag.client, "close_qdrant_client", mock_close_client)
        monkeypatch.setattr(backend.agent.fetch.rag.ingest, "ingest_private_texts", mock_ingest_private)
        monkeypatch.setattr(backend.agent.fetch.rag.config, "QDRANT_PATH", tmp_path / "qdrant")

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

        import backend.agent.fetch.rag.client
        import backend.agent.fetch.rag.ingest
        import backend.agent.fetch.rag.config

        monkeypatch.setattr(backend.agent.fetch.rag.client, "get_qdrant_client", mock_get_client)
        monkeypatch.setattr(backend.agent.fetch.rag.client, "close_qdrant_client", mock_close_client)
        monkeypatch.setattr(backend.agent.fetch.rag.ingest, "ingest_private_texts", mock_ingest_private)
        monkeypatch.setattr(backend.agent.fetch.rag.config, "QDRANT_PATH", tmp_path / "qdrant")

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
            rag_detection_rate=0.92,
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
            rag_detection_rate=0.95,
            avg_relevance=0.90,
            avg_faithfulness=0.92,
        )

        result = print_summary(results, eval_mode="both")
        assert isinstance(result, bool)

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
            rag_detection_rate=0.85,
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

        import backend.agent.fetch.rag.client
        monkeypatch.setattr(backend.agent.fetch.rag.client, "get_qdrant_client", mock_get_client)

        result = check_qdrant_access()
        assert result is True

    def test_check_qdrant_access_locked(self, monkeypatch):
        """Test check_qdrant_access returns False when locked."""
        from backend.eval.output import check_qdrant_access

        def mock_get_client():
            raise Exception("Database already accessed by another process")

        import backend.agent.fetch.rag.client
        monkeypatch.setattr(backend.agent.fetch.rag.client, "get_qdrant_client", mock_get_client)

        result = check_qdrant_access()
        assert result is False

    def test_check_qdrant_access_other_error(self, monkeypatch):
        """Test check_qdrant_access returns True for non-lock errors."""
        from backend.eval.output import check_qdrant_access

        def mock_get_client():
            raise Exception("Connection timeout")

        import backend.agent.fetch.rag.client
        monkeypatch.setattr(backend.agent.fetch.rag.client, "get_qdrant_client", mock_get_client)

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

        import backend.eval.tree
        monkeypatch.setattr(backend.eval.tree, "get_expected_answer", mock_get_expected_answer)

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

        import backend.eval.tree
        monkeypatch.setattr(backend.eval.tree, "get_expected_answer", mock_get_expected_answer)

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

        import backend.eval.tree
        monkeypatch.setattr(backend.eval.tree, "get_expected_answer", mock_get_expected_answer)

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

        import backend.eval.tree
        monkeypatch.setattr(backend.eval.tree, "get_expected_answer", mock_get_expected_answer)

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
                    MockRun("fetch_rag", now, now + timedelta(milliseconds=400)),
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
        import backend.agent.fetch.rag.tools
        import backend.eval.tree
        import backend.eval.runner
        import backend.eval.langsmith

        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.agent.fetch.rag.tools, "tool_entity_rag", mock_tool_entity_rag)
        monkeypatch.setattr(backend.eval.tree, "get_tree_stats", mock_get_tree_stats)
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
        import backend.agent.fetch.rag.tools
        import backend.eval.runner
        import backend.eval.langsmith

        # Must patch in cli module where it's imported
        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.agent.fetch.rag.tools, "tool_entity_rag", mock_tool_entity_rag)
        monkeypatch.setattr(backend.eval.cli, "get_tree_stats", mock_get_tree_stats)
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
        import backend.agent.fetch.rag.tools
        import backend.eval.runner
        import backend.eval.langsmith

        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.agent.fetch.rag.tools, "tool_entity_rag", mock_tool_entity_rag)
        monkeypatch.setattr(backend.eval.cli, "get_tree_stats", mock_get_tree_stats)
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
        import backend.agent.fetch.rag.tools
        import backend.eval.runner

        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.agent.fetch.rag.tools, "tool_entity_rag", mock_tool_entity_rag)
        monkeypatch.setattr(backend.eval.cli, "get_tree_stats", mock_get_tree_stats)
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

        def mock_get_all_paths(role=None):
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

        monkeypatch.setattr(backend.eval.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1, concurrency=1)

        assert results.total_paths == 1
        assert results.paths_passed == 1
        assert results.total_questions == 2

    def test_run_flow_eval_with_failures(self, monkeypatch):
        """Test run_flow_eval with failed paths."""
        from backend.eval.runner import run_flow_eval

        def mock_get_all_paths(role=None):
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

        monkeypatch.setattr(backend.eval.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1, concurrency=1)

        assert results.paths_failed == 1
        assert len(results.failed_paths) == 1

    def test_run_flow_eval_with_rag_metrics(self, monkeypatch):
        """Test run_flow_eval with RAG metrics."""
        from backend.eval.runner import run_flow_eval

        def mock_get_all_paths(role=None):
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

        monkeypatch.setattr(backend.eval.runner, "get_all_paths", mock_get_all_paths)
        monkeypatch.setattr(backend.eval.runner, "test_flow", mock_test_flow)

        results = run_flow_eval(max_paths=1, concurrency=1, eval_mode="rag")

        assert results.avg_account_precision == 0.85
        assert results.avg_account_recall == 0.80
        assert results.account_sample_count == 1

    def test_run_flow_eval_parallel(self, monkeypatch):
        """Test run_flow_eval with parallel execution."""
        from backend.eval.runner import run_flow_eval

        def mock_get_all_paths(role=None):
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

        monkeypatch.setattr(backend.eval.runner, "get_all_paths", mock_get_all_paths)
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


# =============================================================================
# Output Module Extended Tests (coverage for lines 236, 239)
# =============================================================================


class TestCountSloFailures:
    """Tests for _count_slo_failures function."""

    def test_count_slo_failures_sql_queries_failed(self):
        """Test _count_slo_failures counts SQL query failures."""
        from backend.eval.output import _count_slo_failures

        step = FlowStepResult(
            question="Q?",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.95,
            faithfulness_score=0.95,
            answer_correctness_score=0.80,
            sql_queries_total=5,
            sql_queries_success=3,  # 2 failed
        )

        count = _count_slo_failures(step, eval_mode="both")
        assert count >= 1  # Should count SQL query failures

    def test_count_slo_failures_sql_data_validation_failed(self):
        """Test _count_slo_failures counts SQL data validation failures."""
        from backend.eval.output import _count_slo_failures

        step = FlowStepResult(
            question="Q?",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.95,
            faithfulness_score=0.95,
            answer_correctness_score=0.80,
            sql_data_validated=False,  # Failed validation
        )

        count = _count_slo_failures(step, eval_mode="both")
        assert count >= 1  # Should count SQL data validation failure

    def test_count_slo_failures_multiple_failures(self):
        """Test _count_slo_failures counts multiple failure types."""
        from backend.eval.output import _count_slo_failures

        step = FlowStepResult(
            question="Q?",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.5,  # Below SLO
            faithfulness_score=0.5,  # Below SLO
            answer_correctness_score=0.3,  # Below SLO
            sql_queries_total=5,
            sql_queries_success=3,  # Failed
            sql_data_validated=False,  # Failed
            account_rag_invoked=True,
            account_precision_score=0.5,  # Below SLO
            account_recall_score=0.4,  # Below SLO
        )

        count = _count_slo_failures(step, eval_mode="both")
        assert count >= 5  # Multiple failures


# =============================================================================
# Tree Module SQL Validation Tests
# =============================================================================


class TestValidateSqlResults:
    """Tests for validate_sql_results function."""

    def test_validate_sql_results_no_expectations(self):
        """Test validate_sql_results when no expectations defined."""
        from backend.eval.tree import validate_sql_results

        passed, errors = validate_sql_results("Unknown question", {"query": [{"a": 1}]})
        assert passed is True
        assert errors == []

    def test_validate_sql_results_row_count_exact(self, monkeypatch):
        """Test validate_sql_results with exact row_count assertion."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"row_count": 3}},
        )

        # Passing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"a": 1}, {"a": 2}, {"a": 3}]}
        )
        assert passed is True

        # Failing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"a": 1}]}
        )
        assert passed is False
        assert any("row_count" in e for e in errors)

    def test_validate_sql_results_row_count_min(self, monkeypatch):
        """Test validate_sql_results with row_count_min assertion."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"row_count_min": 2}},
        )

        # Passing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"a": 1}, {"a": 2}, {"a": 3}]}
        )
        assert passed is True

        # Failing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"a": 1}]}
        )
        assert passed is False

    def test_validate_sql_results_row_count_max(self, monkeypatch):
        """Test validate_sql_results with row_count_max assertion."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"row_count_max": 2}},
        )

        # Passing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"a": 1}]}
        )
        assert passed is True

        # Failing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"a": 1}, {"a": 2}, {"a": 3}]}
        )
        assert passed is False

    def test_validate_sql_results_total_value(self, monkeypatch):
        """Test validate_sql_results with total_value assertion."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"total_value": 100}},
        )

        # Passing case (within 10% tolerance)
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"value": 50}, {"value": 55}]}
        )
        assert passed is True

        # Failing case (outside tolerance)
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"value": 10}]}
        )
        assert passed is False

    def test_validate_sql_results_must_contain_list(self, monkeypatch):
        """Test validate_sql_results with must_contain list assertion."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"must_contain": ["Acme", "Beta"]}},
        )

        # Passing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Acme Corp"}, {"name": "Beta Inc"}]}
        )
        assert passed is True

        # Failing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Acme Corp"}]}
        )
        assert passed is False

    def test_validate_sql_results_must_contain_single(self, monkeypatch):
        """Test validate_sql_results with must_contain single value."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"must_contain": "Acme"}},
        )

        # Passing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Acme Corp"}]}
        )
        assert passed is True

        # Failing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Other Corp"}]}
        )
        assert passed is False

    def test_validate_sql_results_must_contain_all(self, monkeypatch):
        """Test validate_sql_results with must_contain_all assertion."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"must_contain_all": ["Acme", "Beta", "Gamma"]}},
        )

        # Passing case
        passed, errors = validate_sql_results(
            "Test question?",
            {"query": [{"name": "Acme"}, {"name": "Beta"}, {"name": "Gamma"}]},
        )
        assert passed is True

        # Failing case (missing one)
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Acme"}, {"name": "Beta"}]}
        )
        assert passed is False

    def test_validate_sql_results_must_contain_any(self, monkeypatch):
        """Test validate_sql_results with must_contain_any assertion."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"must_contain_any": ["Acme", "Beta"]}},
        )

        # Passing case (has one)
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Acme Corp"}]}
        )
        assert passed is True

        # Failing case (has neither)
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Other"}]}
        )
        assert passed is False

    def test_validate_sql_results_must_not_contain_list(self, monkeypatch):
        """Test validate_sql_results with must_not_contain list."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"must_not_contain": ["Excluded", "Banned"]}},
        )

        # Passing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Allowed"}]}
        )
        assert passed is True

        # Failing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Excluded Item"}]}
        )
        assert passed is False

    def test_validate_sql_results_must_not_contain_single(self, monkeypatch):
        """Test validate_sql_results with must_not_contain single value."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"must_not_contain": "Excluded"}},
        )

        # Passing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Allowed"}]}
        )
        assert passed is True

        # Failing case
        passed, errors = validate_sql_results(
            "Test question?", {"query": [{"name": "Excluded Item"}]}
        )
        assert passed is False

    def test_validate_sql_results_exact_values(self, monkeypatch):
        """Test validate_sql_results with exact_values assertion."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"exact_values": {"status": ["Active", "Pending"]}}},
        )

        # Passing case
        passed, errors = validate_sql_results(
            "Test question?",
            {"query": [{"status": "Active"}, {"status": "Pending"}]},
        )
        assert passed is True

        # Failing case (different values)
        passed, errors = validate_sql_results(
            "Test question?",
            {"query": [{"status": "Active"}, {"status": "Closed"}]},
        )
        assert passed is False

    def test_validate_sql_results_column_values(self, monkeypatch):
        """Test validate_sql_results with column_values assertion."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"column_values": {"status": ["Active"]}}},
        )

        # Passing case (Active present, extras OK)
        passed, errors = validate_sql_results(
            "Test question?",
            {"query": [{"status": "Active"}, {"status": "Pending"}]},
        )
        assert passed is True

        # Failing case (Active not present)
        passed, errors = validate_sql_results(
            "Test question?",
            {"query": [{"status": "Closed"}]},
        )
        assert passed is False

    def test_validate_sql_results_column_excludes(self, monkeypatch):
        """Test validate_sql_results with column_excludes assertion."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"column_excludes": {"status": ["Excluded"]}}},
        )

        # Passing case
        passed, errors = validate_sql_results(
            "Test question?",
            {"query": [{"status": "Active"}]},
        )
        assert passed is True

        # Failing case
        passed, errors = validate_sql_results(
            "Test question?",
            {"query": [{"status": "Excluded"}]},
        )
        assert passed is False

    def test_validate_sql_results_non_dict_rows(self, monkeypatch):
        """Test validate_sql_results handles non-dict row values."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"must_contain": "test"}},
        )

        # Non-dict rows (list of scalars)
        passed, errors = validate_sql_results(
            "Test question?",
            {"query": ["test value", "other value"]},
        )
        assert passed is True

    def test_validate_sql_results_dict_result(self, monkeypatch):
        """Test validate_sql_results handles dict instead of list."""
        from backend.eval.tree import validate_sql_results
        import backend.eval.tree

        monkeypatch.setattr(
            backend.eval.tree,
            "_EXPECTED_SQL_RESULTS",
            {"Test question?": {"must_contain": "test"}},
        )

        # Single dict instead of list
        passed, errors = validate_sql_results(
            "Test question?",
            {"query": {"name": "test value"}},
        )
        assert passed is True


# =============================================================================
# Tree Module Validation and Print Tests
# =============================================================================


class TestTreeValidation:
    """Tests for tree validation functions."""

    def test_validate_tree(self):
        """Test validate_tree returns list of issues."""
        from backend.eval.tree import validate_tree

        issues = validate_tree()
        assert isinstance(issues, list)

    def test_get_tree_stats(self):
        """Test get_tree_stats returns expected keys."""
        from backend.eval.tree import get_tree_stats

        stats = get_tree_stats()
        assert "num_starters" in stats
        assert "num_questions" in stats
        assert "num_edges" in stats
        assert "num_paths" in stats
        assert "max_depth" in stats
        assert "path_lengths" in stats

    def test_get_all_paths(self):
        """Test get_all_paths returns list of paths."""
        from backend.eval.tree import get_all_paths

        paths = get_all_paths()
        assert isinstance(paths, list)
        if paths:
            assert isinstance(paths[0], list)

    def test_print_tree(self):
        """Test print_tree returns Rich Tree object."""
        from backend.eval.tree import print_tree

        tree = print_tree()
        assert tree is not None

    def test_print_tree_with_max_depth(self):
        """Test print_tree with max_depth parameter."""
        from backend.eval.tree import print_tree

        tree = print_tree(max_depth=2)
        assert tree is not None


# =============================================================================
# Tree Module YAML Loading Tests
# =============================================================================


class TestYamlLoading:
    """Tests for YAML fixture loading."""

    def test_get_expected_answer_exists(self):
        """Test get_expected_answer for existing question."""
        from backend.eval.tree import get_expected_answer

        # May return None if no fixtures, but shouldn't crash
        result = get_expected_answer("What deals are in the pipeline?")
        assert result is None or isinstance(result, str)

    def test_get_expected_answer_not_exists(self):
        """Test get_expected_answer for non-existing question."""
        from backend.eval.tree import get_expected_answer

        result = get_expected_answer("This question does not exist in fixtures")
        assert result is None

    def test_get_expected_rag_exists(self):
        """Test get_expected_rag for existing question."""
        from backend.eval.tree import get_expected_rag

        # May return None if no fixtures
        result = get_expected_rag("What deals are in the pipeline?")
        assert result is None or isinstance(result, bool)

    def test_get_expected_rag_not_exists(self):
        """Test get_expected_rag for non-existing question."""
        from backend.eval.tree import get_expected_rag

        result = get_expected_rag("This question does not exist in fixtures")
        assert result is None

    def test_get_expected_sql_results_not_exists(self):
        """Test get_expected_sql_results for non-existing question."""
        from backend.eval.tree import get_expected_sql_results

        result = get_expected_sql_results("This question does not exist")
        assert result is None


# =============================================================================
# Judge Module Extended Tests
# =============================================================================


class TestJudgeModule:
    """Extended tests for judge module."""

    def test_suppress_event_loop_closed_errors(self):
        """Test _suppress_event_loop_closed_errors doesn't crash."""
        from backend.eval.judge import _suppress_event_loop_closed_errors

        # Should not raise
        _suppress_event_loop_closed_errors()

    def test_is_mock_mode(self, monkeypatch):
        """Test _is_mock_mode returns correct value."""
        from backend.eval.judge import _is_mock_mode

        monkeypatch.setenv("MOCK_LLM", "1")
        # Need to reimport or check current state
        assert _is_mock_mode() is True

        monkeypatch.setenv("MOCK_LLM", "0")
        assert _is_mock_mode() is False


# =============================================================================
# Runner Module Exception Tests
# =============================================================================


class TestRunnerExceptions:
    """Tests for runner exception handling."""

    def test_judge_answer_timeout(self, monkeypatch):
        """Test judge_answer handles timeout."""
        from backend.eval.runner import judge_answer
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        def mock_evaluate_single_timeout(*args, **kwargs):
            raise TimeoutError("Timed out")

        # Must patch in runner module where it's imported
        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "evaluate_single", mock_evaluate_single_timeout)

        result = judge_answer("Q", "A", ["ctx"], timeout=1)
        assert result["ragas_failed"] is True
        assert "timeout" in result["explanation"].lower()

    def test_judge_answer_exception(self, monkeypatch):
        """Test judge_answer handles general exceptions."""
        from backend.eval.runner import judge_answer

        def mock_evaluate_single(*args, **kwargs):
            raise RuntimeError("Evaluation failed")

        # Must patch in runner module where it's imported
        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "evaluate_single", mock_evaluate_single)

        result = judge_answer("Q", "A", ["ctx"])
        assert result["ragas_failed"] is True
        assert "RuntimeError" in result["explanation"] or "Evaluation failed" in result["explanation"]


# =============================================================================
# CLI Module Extended Tests
# =============================================================================


class TestCliModuleExtended:
    """Extended tests for CLI module."""

    def test_route_command(self, monkeypatch):
        """Test route command runs without error."""
        from backend.eval.cli import route

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

        import backend.eval.route.eval
        monkeypatch.setattr(backend.eval.route.eval, "run_sql_eval", mock_run_sql_eval)
        monkeypatch.setattr(backend.eval.route.eval, "print_summary", mock_print_summary)

        # Should not raise
        route(limit=1, verbose=False)

    def test_main_command(self, monkeypatch):
        """Test main command calls _run_eval."""
        from backend.eval.cli import main

        call_args = {}

        def mock_run_eval(**kwargs):
            call_args.update(kwargs)

        import backend.eval.cli
        monkeypatch.setattr(backend.eval.cli, "_run_eval", mock_run_eval)

        main(limit=5, verbose=True, no_judge=True, output="test.json", debug=True, eval_mode="both")

        assert call_args["limit"] == 5
        assert call_args["verbose"] is True
        assert call_args["no_judge"] is True

    def test_run_eval_debug_output(self, monkeypatch, tmp_path):
        """Test _run_eval debug output for failed paths."""
        from backend.eval.cli import _run_eval

        def mock_check_qdrant_access():
            return True

        def mock_tool_entity_rag(*args, **kwargs):
            return "", []

        def mock_get_tree_stats():
            return {"total_paths": 1}

        failed_step = FlowStepResult(
            question="Failed Q with longer question text?",
            answer="Bad A with some content",
            latency_ms=100,
            has_answer=True,
            has_sources=False,
            relevance_score=0.5,
            faithfulness_score=0.5,
            judge_explanation="Low quality answer detected",  # Has explanation
        )
        failed_flow = FlowResult(
            path_id=0,
            questions=["Failed Q with longer question text?"],
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
                all_results=[failed_flow],
            )

        def mock_get_latency_percentages(**kwargs):
            return {}

        import backend.eval.cli
        import backend.agent.fetch.rag.tools
        import backend.eval.runner
        import backend.eval.langsmith

        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.agent.fetch.rag.tools, "tool_entity_rag", mock_tool_entity_rag)
        monkeypatch.setattr(backend.eval.cli, "get_tree_stats", mock_get_tree_stats)
        monkeypatch.setattr(backend.eval.runner, "run_flow_eval", mock_run_flow_eval)
        monkeypatch.setattr(backend.eval.langsmith, "get_latency_percentages", mock_get_latency_percentages)

        # Run with debug=True to trigger debug output path (line 157)
        _run_eval(
            limit=1,
            verbose=False,
            no_judge=True,
            output=None,
            debug=True,
            eval_mode="both",
        )


# =============================================================================
# Tree Module YAML Error Handling Tests
# =============================================================================


class TestYamlLoadingErrors:
    """Tests for YAML loading error handling."""

    def test_load_yaml_fixture_nonexistent_file(self, monkeypatch, tmp_path):
        """Test _load_yaml_fixture with nonexistent file."""
        import backend.eval.tree

        # Temporarily change fixtures path to a nonexistent directory
        monkeypatch.setattr(backend.eval.tree, "_EVAL_FIXTURES_PATH", tmp_path / "nonexistent")

        # Should return empty dict without crashing
        result = backend.eval.tree._load_yaml_fixture("test.yaml")
        assert result == {}

    def test_load_yaml_fixture_invalid_yaml(self, monkeypatch, tmp_path):
        """Test _load_yaml_fixture with invalid YAML content."""
        import backend.eval.tree

        # Create a file with invalid YAML
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("::invalid:: yaml: [content")

        monkeypatch.setattr(backend.eval.tree, "_EVAL_FIXTURES_PATH", tmp_path)

        # Should return empty dict without crashing
        result = backend.eval.tree._load_yaml_fixture("invalid.yaml")
        assert result == {}

    def test_load_yaml_fixture_empty_file(self, monkeypatch, tmp_path):
        """Test _load_yaml_fixture with empty file."""
        import backend.eval.tree

        # Create an empty file
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")

        monkeypatch.setattr(backend.eval.tree, "_EVAL_FIXTURES_PATH", tmp_path)

        # Should return empty dict
        result = backend.eval.tree._load_yaml_fixture("empty.yaml")
        assert result == {}


# =============================================================================
# Tree Module Path Finding Edge Cases
# =============================================================================


class TestTreePathFinding:
    """Tests for tree path finding edge cases."""

    def test_compute_max_depth_no_descendants(self, monkeypatch):
        """Test _compute_max_depth when starters have no descendants."""
        import backend.eval.tree
        import networkx as nx

        # Create a minimal graph with only starters (no descendants)
        mock_g = nx.DiGraph()
        mock_g.add_node("Starter Q?")
        monkeypatch.setattr(backend.eval.tree, "_G", mock_g)
        monkeypatch.setattr(backend.eval.tree, "_STARTERS", ["Starter Q?"])

        result = backend.eval.tree._compute_max_depth(["Starter Q?"])
        assert result == 0

    def test_find_paths_with_nx_no_path(self, monkeypatch):
        """Test _find_paths handles NetworkXNoPath exception."""
        import backend.eval.tree
        import networkx as nx

        # Create a simple graph with a path
        mock_g = nx.DiGraph()
        mock_g.add_edge("Starter?", "Child?")
        mock_g.add_edge("Child?", "Leaf?")

        monkeypatch.setattr(backend.eval.tree, "_G", mock_g)

        # Create subgraph with all nodes
        subgraph = mock_g.subgraph(["Starter?", "Child?", "Leaf?"])

        # This should work - leaf is Leaf?
        result = backend.eval.tree._find_paths(["Starter?"], subgraph, 5)
        assert len(result) > 0  # Should find paths


# =============================================================================
# Tree Validation Edge Cases
# =============================================================================


class TestTreeValidationEdgeCases:
    """Tests for tree validation edge cases."""

    def test_validate_tree_starter_not_in_graph(self, monkeypatch):
        """Test validate_tree when starter is not in graph."""
        import backend.eval.tree
        import networkx as nx

        # Create empty graph but have starters defined
        mock_g = nx.DiGraph()
        monkeypatch.setattr(backend.eval.tree, "_G", mock_g)
        monkeypatch.setattr(backend.eval.tree, "_STARTERS", ["Missing Starter?"])

        issues = backend.eval.tree.validate_tree()
        assert any("Starter not in tree" in issue for issue in issues)

    def test_validate_tree_orphaned_question(self, monkeypatch):
        """Test validate_tree detects orphaned questions."""
        import backend.eval.tree
        import networkx as nx

        # Create graph with connected and orphaned nodes
        mock_g = nx.DiGraph()
        mock_g.add_edge("Starter?", "Connected?")
        mock_g.add_node("Orphan?")  # Not reachable from starter
        monkeypatch.setattr(backend.eval.tree, "_G", mock_g)
        monkeypatch.setattr(backend.eval.tree, "_STARTERS", ["Starter?"])

        issues = backend.eval.tree.validate_tree()
        assert any("Orphaned question" in issue for issue in issues)

    def test_validate_tree_cycle_detection(self, monkeypatch):
        """Test validate_tree detects cycles."""
        import backend.eval.tree
        import networkx as nx

        # Create graph with a cycle
        mock_g = nx.DiGraph()
        mock_g.add_edge("A?", "B?")
        mock_g.add_edge("B?", "C?")
        mock_g.add_edge("C?", "A?")  # Creates cycle
        monkeypatch.setattr(backend.eval.tree, "_G", mock_g)
        monkeypatch.setattr(backend.eval.tree, "_STARTERS", ["A?"])

        issues = backend.eval.tree.validate_tree()
        assert any("cycles" in issue.lower() for issue in issues)

    def test_validate_tree_wrong_out_degree(self, monkeypatch):
        """Test validate_tree detects wrong number of follow-ups."""
        import backend.eval.tree
        import networkx as nx

        # Create graph where node has 2 follow-ups (not 0 or 3)
        mock_g = nx.DiGraph()
        mock_g.add_edge("Starter?", "Child1?")
        mock_g.add_edge("Starter?", "Child2?")  # Only 2 children, not 0 or 3
        monkeypatch.setattr(backend.eval.tree, "_G", mock_g)
        monkeypatch.setattr(backend.eval.tree, "_STARTERS", ["Starter?"])

        issues = backend.eval.tree.validate_tree()
        assert any("follow-ups" in issue for issue in issues)


# =============================================================================
# Route Eval Module Tests
# =============================================================================


class TestRouteEvalModule:
    """Tests for route evaluation module."""

    def test_route_eval_run_sql_eval(self, monkeypatch):
        """Test run_sql_eval function exists and runs."""
        from backend.eval.route.eval import run_sql_eval, EvalResults

        # Mock to return empty list of questions
        def mock_load_test_questions():
            return []

        import backend.eval.route.eval
        monkeypatch.setattr(backend.eval.route.eval, "_load_test_questions", mock_load_test_questions)

        # Run with empty questions list
        results = run_sql_eval(limit=0, verbose=False)
        assert isinstance(results, EvalResults)

    def test_route_eval_print_summary(self, capsys):
        """Test print_summary function."""
        from backend.eval.route.eval import print_summary, EvalResults

        results = EvalResults(
            total=10,
            passed=8,
            sql_executed=10,
            sql_failed=0,
            cases=[],
        )

        print_summary(results)
        # Should not raise

    def test_route_eval_print_summary_with_failures(self, capsys):
        """Test print_summary with failed cases."""
        from backend.eval.route.eval import print_summary, EvalResults, CaseResult

        failed_case = CaseResult(
            question="Test question?",
            sql="SELECT * FROM test",
            passed=False,
            row_count=0,
            errors=["Expected 5 rows, got 0"],
        )

        results = EvalResults(
            total=1,
            passed=0,
            sql_executed=1,
            sql_failed=0,
            cases=[failed_case],
        )

        print_summary(results)
        # Should print failure details


# =============================================================================
# Output Print SLO Failures Tests
# =============================================================================


class TestPrintSloFailuresExtended:
    """Extended tests for _print_slo_failures function."""

    def test_print_slo_failures_with_multiple_failures(self, capsys):
        """Test _print_slo_failures shows failures sorted by severity."""
        from backend.eval.output import _print_slo_failures

        # Create steps with varying failure counts
        step1 = FlowStepResult(
            question="Q1?",
            answer="A1",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.5,  # Fails
            faithfulness_score=0.5,  # Fails
            answer_correctness_score=0.3,  # Fails
        )
        step2 = FlowStepResult(
            question="Q2?",
            answer="A2",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.5,  # Fails
            faithfulness_score=0.9,
            answer_correctness_score=0.8,
        )

        flow1 = FlowResult(
            path_id=0,
            questions=["Q1?"],
            steps=[step1],
            total_latency_ms=100,
            success=False,
        )
        flow2 = FlowResult(
            path_id=1,
            questions=["Q2?"],
            steps=[step2],
            total_latency_ms=100,
            success=False,
        )

        results = FlowEvalResults(
            total_paths=2,
            paths_tested=2,
            paths_passed=0,
            paths_failed=2,
            total_questions=2,
            questions_passed=0,
            questions_failed=2,
            all_results=[flow1, flow2],
        )

        # Should print failures table without crashing
        _print_slo_failures(results, eval_mode="both")

    def test_print_slo_failures_rag_only_mode(self, capsys):
        """Test _print_slo_failures in rag-only eval mode."""
        from backend.eval.output import _print_slo_failures

        step = FlowStepResult(
            question="Q?",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.9,
            faithfulness_score=0.9,
            account_rag_invoked=True,
            account_precision_score=0.5,  # Fails
            account_recall_score=0.4,  # Fails
        )

        flow = FlowResult(
            path_id=0,
            questions=["Q?"],
            steps=[step],
            total_latency_ms=100,
            success=False,
        )

        results = FlowEvalResults(
            total_paths=1,
            paths_tested=1,
            paths_passed=0,
            paths_failed=1,
            total_questions=1,
            questions_passed=0,
            questions_failed=1,
            all_results=[flow],
        )

        # Should print with RAG columns only
        _print_slo_failures(results, eval_mode="rag")

    def test_print_slo_failures_pipeline_only_mode(self, capsys):
        """Test _print_slo_failures in pipeline-only eval mode."""
        from backend.eval.output import _print_slo_failures

        step = FlowStepResult(
            question="Q?",
            answer="A",
            latency_ms=100,
            has_answer=True,
            has_sources=True,
            relevance_score=0.5,  # Fails
            faithfulness_score=0.5,  # Fails
            answer_correctness_score=0.3,  # Fails
        )

        flow = FlowResult(
            path_id=0,
            questions=["Q?"],
            steps=[step],
            total_latency_ms=100,
            success=False,
        )

        results = FlowEvalResults(
            total_paths=1,
            paths_tested=1,
            paths_passed=0,
            paths_failed=1,
            total_questions=1,
            questions_passed=0,
            questions_failed=1,
            all_results=[flow],
        )

        # Should print with pipeline columns only
        _print_slo_failures(results, eval_mode="pipeline")


# =============================================================================
# Runner Module Extended Coverage Tests
# =============================================================================


class TestRunnerEdgeCases:
    """Extended tests for runner edge cases."""

    def test_test_single_question_with_sql_data_validation(self, monkeypatch):
        """Test test_single_question with SQL data validation."""
        from backend.eval.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return {
                "answer": "Test answer with good content",
                "sources": [],
                "sql_results": {"query": [{"company": "Acme", "value": 100}]},
                "account_chunks": [],
                "needs_rag": False,
            }

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

        def mock_get_expected_answer(question):
            return "Expected answer"

        def mock_get_expected_sql_results(question):
            return {"must_contain": "Acme"}

        import backend.eval.runner

        monkeypatch.setattr(backend.eval.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.runner, "judge_answer", mock_judge_answer)
        # Patch in runner module where functions are imported
        monkeypatch.setattr(backend.eval.runner, "get_expected_answer", mock_get_expected_answer)
        monkeypatch.setattr(backend.eval.runner, "get_expected_sql_results", mock_get_expected_sql_results)

        result = test_single_question("Test Q?", [], "session1")

        assert result.sql_data_validated is True
        assert result.sql_data_errors is None

    def test_test_single_question_ragas_failed_rag_mode(self, monkeypatch):
        """Test test_single_question with RAGAS failure in RAG mode."""
        from backend.eval.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return {
                "answer": "Test answer",
                "sources": [],
                "sql_results": {},
                "account_chunks": ["Some context"],
                "account_rag_invoked": True,  # Must be True for RAG metrics to be evaluated
                "needs_rag": True,
            }

        def mock_judge_answer(*args, **kwargs):
            return {
                "relevance": 0.0,
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "explanation": "RAGAS failed",
                "ragas_failed": True,  # RAGAS failure
                "nan_metrics": ["context_precision", "context_recall"],
            }

        def mock_get_expected_answer(question):
            return "Expected"  # Need reference for RAG metrics

        import backend.eval.runner

        monkeypatch.setattr(backend.eval.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.runner, "judge_answer", mock_judge_answer)
        monkeypatch.setattr(backend.eval.runner, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Q?", [], "session1", eval_mode="rag")

        assert result.ragas_metrics_failed >= 2

    def test_test_single_question_ragas_failed_pipeline_mode(self, monkeypatch):
        """Test test_single_question with RAGAS failure in pipeline mode."""
        from backend.eval.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return {
                "answer": "Test answer",
                "sources": [],
                "sql_results": {"query": [{"a": 1}]},
                "account_chunks": [],
                "needs_rag": False,
            }

        def mock_judge_answer(*args, **kwargs):
            return {
                "relevance": 0.0,
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "explanation": "RAGAS failed",
                "ragas_failed": True,  # RAGAS failure
                "nan_metrics": [],
            }

        def mock_get_expected_answer(question):
            return None

        import backend.eval.runner
        import backend.eval.tree

        monkeypatch.setattr(backend.eval.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.runner, "judge_answer", mock_judge_answer)
        monkeypatch.setattr(backend.eval.tree, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Q?", [], "session1", eval_mode="pipeline")

        assert result.ragas_metrics_failed >= 3

    def test_test_single_question_ragas_failed_both_mode(self, monkeypatch):
        """Test test_single_question with RAGAS failure in both mode."""
        from backend.eval.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            return {
                "answer": "Test answer",
                "sources": [],
                "sql_results": {"query": [{"a": 1}]},
                "account_chunks": ["context"],
                "account_rag_invoked": True,  # Must be True for RAG metrics to be evaluated
                "needs_rag": True,
            }

        call_count = {"count": 0}

        def mock_judge_answer(*args, **kwargs):
            call_count["count"] += 1
            return {
                "relevance": 0.0,
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "explanation": "RAGAS failed",
                "ragas_failed": True,  # RAGAS failure
                "nan_metrics": [],
            }

        def mock_get_expected_answer(question):
            return "Expected"

        import backend.eval.runner
        import backend.eval.tree

        monkeypatch.setattr(backend.eval.runner, "_invoke_agent", mock_invoke_agent)
        monkeypatch.setattr(backend.eval.runner, "judge_answer", mock_judge_answer)
        monkeypatch.setattr(backend.eval.tree, "get_expected_answer", mock_get_expected_answer)

        result = test_single_question("Q?", [], "session1", eval_mode="both")

        # Should have failed metrics from both RAG and pipeline evaluations
        assert result.ragas_metrics_failed >= 2


# =============================================================================
# Route Eval Extended Tests
# =============================================================================


class TestRouteEvalExtended:
    """Extended tests for route evaluation module."""

    def test_route_eval_run_with_verbose(self, monkeypatch, tmp_path):
        """Test run_sql_eval with verbose output and actual execution."""
        from backend.eval.route.eval import run_sql_eval, EvalResults

        # Create mock fixtures file
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        (fixtures_dir / "expected_sql_results.yaml").write_text(
            "Test question?:\n  must_contain: test\n"
        )

        def mock_load_test_questions():
            return ["Test question?"]

        def mock_get_sql_plan(question):
            from backend.agent.fetch.planner import SQLPlan
            return SQLPlan(sql="SELECT 'test' as value", needs_rag=False)

        class MockResult:
            description = [("value",)]
            def fetchall(self):
                return [("test",)]

        class MockConnection:
            def execute(self, sql):
                return MockResult()

        def mock_get_connection():
            return MockConnection()

        import backend.eval.route.eval
        import backend.agent.fetch.planner
        import backend.agent.fetch.sql.connection

        monkeypatch.setattr(backend.eval.route.eval, "_load_test_questions", mock_load_test_questions)
        monkeypatch.setattr(backend.eval.route.eval, "get_sql_plan", mock_get_sql_plan)
        monkeypatch.setattr(backend.eval.route.eval, "get_connection", mock_get_connection)

        results = run_sql_eval(verbose=True, limit=1)
        assert isinstance(results, EvalResults)
        assert results.total == 1

    def test_route_eval_sql_execution_error(self, monkeypatch):
        """Test run_sql_eval handles SQL execution errors."""
        from backend.eval.route.eval import run_sql_eval, EvalResults

        def mock_load_test_questions():
            return ["Bad SQL question?"]

        def mock_get_sql_plan(question):
            from backend.agent.fetch.planner import SQLPlan
            return SQLPlan(sql="INVALID SQL", needs_rag=False)

        class MockConnection:
            def execute(self, sql):
                raise Exception("SQL syntax error")

        def mock_get_connection():
            return MockConnection()

        import backend.eval.route.eval

        monkeypatch.setattr(backend.eval.route.eval, "_load_test_questions", mock_load_test_questions)
        monkeypatch.setattr(backend.eval.route.eval, "get_sql_plan", mock_get_sql_plan)
        monkeypatch.setattr(backend.eval.route.eval, "get_connection", mock_get_connection)

        results = run_sql_eval(verbose=True, limit=1)
        assert results.sql_failed == 1

    def test_route_eval_planner_error(self, monkeypatch):
        """Test run_sql_eval handles planner errors."""
        from backend.eval.route.eval import run_sql_eval, EvalResults

        def mock_load_test_questions():
            return ["Error question?"]

        def mock_get_sql_plan(question):
            raise Exception("Planner crashed")

        import backend.eval.route.eval

        monkeypatch.setattr(backend.eval.route.eval, "_load_test_questions", mock_load_test_questions)
        monkeypatch.setattr(backend.eval.route.eval, "get_sql_plan", mock_get_sql_plan)

        results = run_sql_eval(verbose=True, limit=1)
        assert results.passed == 0  # Should fail

    def test_route_eval_no_assertions(self, monkeypatch):
        """Test run_sql_eval when no assertions defined for question."""
        from backend.eval.route.eval import run_sql_eval, EvalResults

        def mock_load_test_questions():
            return ["Unknown question?"]

        def mock_get_sql_plan(question):
            from backend.agent.fetch.planner import SQLPlan
            return SQLPlan(sql="SELECT 1 as value", needs_rag=False)

        class MockResult:
            description = [("value",)]
            def fetchall(self):
                return [(1,)]

        class MockConnection:
            def execute(self, sql):
                return MockResult()

        def mock_get_connection():
            return MockConnection()

        def mock_get_expected_sql_results(question):
            return None  # No assertions

        import backend.eval.route.eval
        import backend.eval.tree

        monkeypatch.setattr(backend.eval.route.eval, "_load_test_questions", mock_load_test_questions)
        monkeypatch.setattr(backend.eval.route.eval, "get_sql_plan", mock_get_sql_plan)
        monkeypatch.setattr(backend.eval.route.eval, "get_connection", mock_get_connection)
        monkeypatch.setattr(backend.eval.tree, "get_expected_sql_results", mock_get_expected_sql_results)

        results = run_sql_eval(verbose=False, limit=1)
        # Should pass if results returned (no assertions)
        assert results.total == 1


# =============================================================================
# Tree Network Path Edge Cases
# =============================================================================


class TestTreeNetworkPaths:
    """Test tree path finding with network exceptions."""

    def test_compute_max_depth_no_path_between_nodes(self, monkeypatch):
        """Test _compute_max_depth when no path exists between starter and descendant."""
        import networkx as nx
        import backend.eval.tree as tree_module

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
        import backend.eval.tree as tree_module

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
# Runner Timeout and Concurrent Execution Tests
# =============================================================================


class TestRunnerTimeoutHandling:
    """Test runner timeout and concurrent execution edge cases."""

    def test_test_single_question_timeout_error(self, monkeypatch):
        """Test test_single_question handles TimeoutError."""
        from backend.eval.runner import test_single_question

        def mock_invoke_agent(question, session_id=None):
            raise TimeoutError("Request timed out")

        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "_invoke_agent", mock_invoke_agent)

        result = test_single_question("Q?", [], "session1")

        assert result.has_answer is False
        assert "Timeout" in result.error

    def test_run_flow_eval_concurrent_exception(self, monkeypatch):
        """Test run_flow_eval handles exceptions in concurrent futures."""
        from backend.eval.runner import run_flow_eval
        from backend.eval.models import FlowResult, FlowStepResult

        call_count = {"count": 0}

        def mock_test_flow(path, path_id, use_judge, verbose, eval_mode):
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise RuntimeError("Thread crashed unexpectedly")
            return FlowResult(
                path_id=path_id,
                questions=path,
                steps=[FlowStepResult(
                    question="Q?", answer="A", latency_ms=100,
                    has_answer=True, has_sources=True,
                    relevance_score=0.9, faithfulness_score=0.9,
                )],
                total_latency_ms=100,
                success=True,
            )

        def mock_get_all_paths():
            return [["Q1?"], ["Q2?"]]  # Two paths

        import backend.eval.runner
        monkeypatch.setattr(backend.eval.runner, "test_flow", mock_test_flow)
        monkeypatch.setattr(backend.eval.runner, "get_all_paths", mock_get_all_paths)

        # Run with concurrency=2 to trigger ThreadPoolExecutor path
        results = run_flow_eval(max_paths=2, concurrency=2)

        # Should have handled the exception gracefully
        assert results.paths_tested == 2


# =============================================================================
# CLI Exception Handling Tests
# =============================================================================


class TestCliExceptionHandling:
    """Test CLI handles exceptions gracefully."""

    def test_run_eval_flow_eval_exception(self, monkeypatch):
        """Test _run_eval handles exceptions from run_flow_eval."""
        from backend.eval.cli import _run_eval

        def mock_check_qdrant_access():
            return True

        def mock_ensure_collections():
            pass

        def mock_run_flow_eval(**kwargs):
            raise RuntimeError("Flow eval crashed")

        def mock_get_latency_percentages(**kwargs):
            return None

        def mock_get_tree_stats():
            return {"num_starters": 1, "num_questions": 5, "num_edges": 4, "max_depth": 3, "num_paths": 2, "path_lengths": {"min": 2, "max": 3}}

        import backend.eval.cli

        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.eval.cli, "ensure_qdrant_collections", mock_ensure_collections)
        monkeypatch.setattr(backend.eval.cli, "run_flow_eval", mock_run_flow_eval)
        monkeypatch.setattr(backend.eval.cli, "get_latency_percentages", mock_get_latency_percentages)
        monkeypatch.setattr(backend.eval.cli, "get_tree_stats", mock_get_tree_stats)

        # Should not raise, but handle gracefully
        _run_eval(limit=1, verbose=False, no_judge=True, output=None, debug=False, eval_mode="both")

    def test_run_eval_debug_with_judge_explanation(self, monkeypatch, capsys):
        """Test _run_eval debug output includes judge explanation."""
        from backend.eval.cli import _run_eval
        from backend.eval.models import FlowEvalResults, FlowResult, FlowStepResult

        def mock_check_qdrant_access():
            return True

        def mock_ensure_collections():
            pass

        failed_step = FlowStepResult(
            question="Failed Q?",
            answer="Bad answer",
            latency_ms=100,
            has_answer=True,
            has_sources=False,
            relevance_score=0.3,
            faithfulness_score=0.3,
            judge_explanation="This answer is not relevant to the question.",
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
                failed_paths=[
                    FlowResult(
                        path_id=1,
                        questions=["Failed Q?"],
                        steps=[failed_step],
                        total_latency_ms=100,
                        success=False,
                    )
                ],
                all_results=[],
            )

        def mock_get_latency_percentages(**kwargs):
            return None

        def mock_print_summary(*args, **kwargs):
            pass

        def mock_get_tree_stats():
            return {"num_starters": 1, "num_questions": 5, "num_edges": 4, "max_depth": 3, "num_paths": 2, "path_lengths": {"min": 2, "max": 3}}

        import backend.eval.cli

        monkeypatch.setattr(backend.eval.cli, "check_qdrant_access", mock_check_qdrant_access)
        monkeypatch.setattr(backend.eval.cli, "ensure_qdrant_collections", mock_ensure_collections)
        monkeypatch.setattr(backend.eval.cli, "run_flow_eval", mock_run_flow_eval)
        monkeypatch.setattr(backend.eval.cli, "get_latency_percentages", mock_get_latency_percentages)
        monkeypatch.setattr(backend.eval.cli, "print_summary", mock_print_summary)
        monkeypatch.setattr(backend.eval.cli, "get_tree_stats", mock_get_tree_stats)

        # Run with debug=True
        _run_eval(limit=1, verbose=False, no_judge=True, output=None, debug=True, eval_mode="both")


# =============================================================================
# Route Eval Verbose Output Tests
# =============================================================================


class TestRouteEvalVerboseOutput:
    """Test route eval verbose output paths."""

    def test_run_sql_eval_verbose_with_errors(self, monkeypatch):
        """Test run_sql_eval verbose output with validation errors."""
        from backend.eval.route.eval import run_sql_eval

        def mock_load_test_questions():
            return ["Question with validation errors?"]

        def mock_get_sql_plan(question):
            from backend.agent.fetch.planner import SQLPlan
            return SQLPlan(sql="SELECT 'test' as name", needs_rag=False)

        class MockResult:
            description = [("name",)]
            def fetchall(self):
                return [("test",)]

        class MockConnection:
            def execute(self, sql):
                return MockResult()

        def mock_get_connection():
            return MockConnection()

        def mock_validate(question, results):
            return False, ["row_count mismatch: expected 5, got 1"]

        def mock_get_expected_sql_results(question):
            return {"row_count": 5}  # Will cause mismatch

        import backend.eval.route.eval
        import backend.eval.tree

        monkeypatch.setattr(backend.eval.route.eval, "_load_test_questions", mock_load_test_questions)
        monkeypatch.setattr(backend.eval.route.eval, "get_sql_plan", mock_get_sql_plan)
        monkeypatch.setattr(backend.eval.route.eval, "get_connection", mock_get_connection)
        monkeypatch.setattr(backend.eval.route.eval, "validate_sql_results", mock_validate)
        monkeypatch.setattr(backend.eval.route.eval, "get_expected_sql_results", mock_get_expected_sql_results)

        # Run with verbose=True to trigger error output
        results = run_sql_eval(verbose=True, limit=1)

        assert results.passed == 0
        assert len(results.cases) == 1
        assert len(results.cases[0].errors) > 0
