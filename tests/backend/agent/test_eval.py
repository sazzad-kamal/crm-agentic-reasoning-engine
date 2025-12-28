"""
Tests for backend.agent.eval module.

Tests the agent evaluation models, tracking, and history functions.
"""

import os

import pytest

# Set mock mode
os.environ["MOCK_LLM"] = "1"

from backend.agent.eval.models import (
    ToolEvalResult,
    RouterEvalResult,
    E2EEvalResult,
    ToolEvalSummary,
    RouterEvalSummary,
    E2EEvalSummary,
    AgentEvalSummary,
    SLO_LATENCY_P95_MS,
    SLO_TOOL_ACCURACY,
    SLO_ROUTER_ACCURACY,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
)


# =============================================================================
# Model Tests
# =============================================================================

class TestToolEvalResult:
    """Tests for ToolEvalResult model."""

    def test_tool_eval_result_creation(self):
        """Test creating a ToolEvalResult."""
        result = ToolEvalResult(
            tool_name="tool_company_lookup",
            test_case_id="tc1",
            input_params={"company_id": "ACME-MFG"},
            expected_found=True,
            actual_found=True,
            expected_company_id="ACME-MFG",
            actual_company_id="ACME-MFG",
            data_correct=True,
            sources_present=True,
            latency_ms=50.0,
        )

        assert result.tool_name == "tool_company_lookup"
        assert result.expected_found is True
        assert result.actual_found is True
        assert result.data_correct is True

    def test_tool_eval_result_with_error(self):
        """Test ToolEvalResult with error."""
        result = ToolEvalResult(
            tool_name="tool_company_lookup",
            test_case_id="tc1",
            input_params={"company_id": "INVALID"},
            expected_found=True,
            actual_found=False,
            data_correct=False,
            sources_present=False,
            error="Company not found",
        )

        assert result.actual_found is False
        assert result.error == "Company not found"


class TestRouterEvalResult:
    """Tests for RouterEvalResult model."""

    def test_router_eval_result_creation(self):
        """Test creating a RouterEvalResult."""
        result = RouterEvalResult(
            test_case_id="r1",
            question="What is going on with Acme?",
            expected_mode="data",
            actual_mode="data",
            expected_company_id="ACME-MFG",
            actual_company_id="ACME-MFG",
            mode_correct=True,
            company_correct=True,
            intent_expected="company_status",
            intent_actual="company_status",
            intent_correct=True,
        )

        assert result.mode_correct is True
        assert result.company_correct is True
        assert result.intent_correct is True

    def test_router_eval_result_mode_mismatch(self):
        """Test RouterEvalResult with mode mismatch."""
        result = RouterEvalResult(
            test_case_id="r1",
            question="How do I import contacts?",
            expected_mode="docs",
            actual_mode="data",
            mode_correct=False,
            company_correct=True,
        )

        assert result.mode_correct is False


class TestE2EEvalResult:
    """Tests for E2EEvalResult model."""

    def test_e2e_eval_result_creation(self):
        """Test creating an E2EEvalResult."""
        result = E2EEvalResult(
            test_case_id="e2e1",
            question="What's the status of Acme Manufacturing?",
            category="company_status",
            expected_company_id="ACME-MFG",
            actual_company_id="ACME-MFG",
            company_correct=True,
            expected_intent="company_status",
            actual_intent="company_status",
            intent_correct=True,
            answer="Acme Manufacturing is doing well.",
            answer_relevance=1,
            answer_grounded=1,
            has_sources=True,
            latency_ms=200.0,
            total_tokens=500,
        )

        assert result.answer_relevance == 1
        assert result.answer_grounded == 1
        assert result.intent_correct is True

    def test_e2e_eval_result_with_judge_explanation(self):
        """Test E2EEvalResult with judge explanation."""
        result = E2EEvalResult(
            test_case_id="e2e1",
            question="Test question",
            category="test",
            expected_intent="general",
            actual_intent="general",
            intent_correct=True,
            answer="Test answer",
            answer_relevance=0,
            answer_grounded=0,
            has_sources=False,
            latency_ms=100.0,
            total_tokens=100,
            judge_explanation="Answer was not relevant to the question.",
        )

        assert result.judge_explanation == "Answer was not relevant to the question."


class TestToolEvalSummary:
    """Tests for ToolEvalSummary model."""

    def test_tool_eval_summary_creation(self):
        """Test creating a ToolEvalSummary."""
        summary = ToolEvalSummary(
            total_tests=20,
            passed=18,
            failed=2,
            accuracy=0.90,
            by_tool={
                "tool_company_lookup": {"passed": 10, "failed": 0, "accuracy": 1.0},
                "tool_activities": {"passed": 8, "failed": 2, "accuracy": 0.8},
            },
        )

        assert summary.total_tests == 20
        assert summary.accuracy == 0.90
        assert summary.by_tool["tool_company_lookup"]["accuracy"] == 1.0


class TestRouterEvalSummary:
    """Tests for RouterEvalSummary model."""

    def test_router_eval_summary_creation(self):
        """Test creating a RouterEvalSummary."""
        summary = RouterEvalSummary(
            total_tests=30,
            mode_accuracy=0.93,
            company_extraction_accuracy=0.97,
            intent_accuracy=0.90,
            by_mode={
                "data": {"expected": 15, "correct": 14, "accuracy": 0.93},
                "docs": {"expected": 15, "correct": 14, "accuracy": 0.93},
            },
        )

        assert summary.mode_accuracy == 0.93
        assert summary.intent_accuracy == 0.90


class TestE2EEvalSummary:
    """Tests for E2EEvalSummary model."""

    def test_e2e_eval_summary_creation(self):
        """Test creating an E2EEvalSummary."""
        summary = E2EEvalSummary(
            total_tests=25,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.92,
            answer_relevance_rate=0.88,
            groundedness_rate=0.84,
            avg_latency_ms=350.0,
            p95_latency_ms=800.0,
            latency_slo_pass=True,
            by_category={
                "company_status": {"count": 10, "relevance": 0.9, "grounded": 0.8},
                "docs_query": {"count": 15, "relevance": 0.87, "grounded": 0.87},
            },
        )

        assert summary.answer_relevance_rate == 0.88
        assert summary.latency_slo_pass is True


class TestAgentEvalSummary:
    """Tests for AgentEvalSummary model."""

    def test_agent_eval_summary_creation(self):
        """Test creating an AgentEvalSummary."""
        e2e = E2EEvalSummary(
            total_tests=25,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.92,
            answer_relevance_rate=0.88,
            groundedness_rate=0.84,
            avg_latency_ms=350.0,
            p95_latency_ms=800.0,
            by_category={},
        )

        summary = AgentEvalSummary(
            e2e_eval=e2e,
            overall_score=0.88,
            all_slos_passed=True,
        )

        assert summary.e2e_eval is not None
        assert summary.overall_score == 0.88
        assert summary.all_slos_passed is True


# =============================================================================
# SLO Constants Tests
# =============================================================================

class TestAgentSLOConstants:
    """Tests for agent SLO constant values."""

    def test_slo_latency(self):
        """Test SLO latency threshold."""
        assert SLO_LATENCY_P95_MS == 5000

    def test_slo_tool_accuracy(self):
        """Test SLO tool accuracy threshold."""
        assert SLO_TOOL_ACCURACY == 0.90

    def test_slo_router_accuracy(self):
        """Test SLO router accuracy threshold."""
        assert SLO_ROUTER_ACCURACY == 0.90

    def test_slo_answer_relevance(self):
        """Test SLO answer relevance threshold."""
        assert SLO_ANSWER_RELEVANCE == 0.80

    def test_slo_groundedness(self):
        """Test SLO groundedness threshold."""
        assert SLO_GROUNDEDNESS == 0.80


# =============================================================================
# Agent Tracking Module Tests
# =============================================================================

class TestAgentTrackingModule:
    """Tests for the agent tracking module functions."""

    def test_compare_e2e_with_previous_no_previous(self):
        """Test comparison when no previous run exists."""
        from backend.agent.eval.tracking import compare_e2e_with_previous

        current = E2EEvalSummary(
            total_tests=20,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.90,
            answer_relevance_rate=0.85,
            groundedness_rate=0.80,
            avg_latency_ms=400.0,
            p95_latency_ms=1000.0,
            by_category={},
        )

        comparison = compare_e2e_with_previous(current, None)

        assert comparison["has_previous"] is False
        assert comparison["regressions"] == []
        assert comparison["improvements"] == []

    def test_compare_e2e_with_previous_detects_regression(self):
        """Test that comparison detects regressions."""
        from backend.agent.eval.tracking import compare_e2e_with_previous

        previous = E2EEvalSummary(
            total_tests=20,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.95,
            answer_relevance_rate=0.90,
            groundedness_rate=0.90,
            avg_latency_ms=300.0,
            p95_latency_ms=800.0,
            by_category={},
        )

        current = E2EEvalSummary(
            total_tests=20,
            company_extraction_accuracy=0.95,
            intent_accuracy=0.95,
            answer_relevance_rate=0.70,  # Regression
            groundedness_rate=0.90,
            avg_latency_ms=300.0,
            p95_latency_ms=800.0,
            by_category={},
        )

        comparison = compare_e2e_with_previous(current, previous)

        assert comparison["has_previous"] is True
        assert len(comparison["regressions"]) >= 1

    def test_analyze_e2e_budget_violations_no_violations(self):
        """Test budget analysis with no violations."""
        from backend.agent.eval.tracking import analyze_e2e_budget_violations

        results = [
            E2EEvalResult(
                test_case_id="t1",
                question="Test",
                category="test",
                expected_intent="general",
                actual_intent="general",
                intent_correct=True,
                answer="Test",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=1000.0,  # Well under budget
                total_tokens=100,
            )
        ]

        analysis = analyze_e2e_budget_violations(results)

        assert len(analysis["total_violations"]) == 0

    def test_analyze_e2e_budget_violations_with_violations(self):
        """Test budget analysis with violations."""
        from backend.agent.eval.tracking import analyze_e2e_budget_violations

        results = [
            E2EEvalResult(
                test_case_id="t1",
                question="Test",
                category="test",
                expected_intent="general",
                actual_intent="general",
                intent_correct=True,
                answer="Test",
                answer_relevance=1,
                answer_grounded=1,
                has_sources=True,
                latency_ms=6000.0,  # Over budget (5000ms)
                total_tokens=100,
            )
        ]

        analysis = analyze_e2e_budget_violations(results)

        assert len(analysis["total_violations"]) == 1
        assert analysis["total_violations"][0]["test_case_id"] == "t1"


# =============================================================================
# Agent History Module Tests
# =============================================================================

class TestAgentHistoryModule:
    """Tests for the agent history module functions."""

    def test_compute_agent_trends_insufficient_data(self):
        """Test trend computation with insufficient data."""
        from backend.agent.eval.history import compute_agent_trends

        history = [{"metrics": {"answer_relevance": 0.8}}]
        result = compute_agent_trends(history, "answer_relevance")

        assert result["has_trend"] is False

    def test_compute_agent_trends_with_data(self):
        """Test trend computation with sufficient data."""
        from backend.agent.eval.history import compute_agent_trends

        history = [
            {"metrics": {"answer_relevance": 0.70}},
            {"metrics": {"answer_relevance": 0.75}},
            {"metrics": {"answer_relevance": 0.85}},
        ]

        result = compute_agent_trends(history, "answer_relevance")

        assert result["has_trend"] is True
        assert result["min"] == 0.70
        assert result["max"] == 0.85
        assert result["current"] == 0.85

    def test_compute_agent_trends_direction(self):
        """Test trend direction computation."""
        from backend.agent.eval.history import compute_agent_trends

        # Upward trend
        history_up = [
            {"metrics": {"score": 0.60}},
            {"metrics": {"score": 0.85}},
        ]
        result = compute_agent_trends(history_up, "score")
        assert result["trend_direction"] == "up"

        # Downward trend
        history_down = [
            {"metrics": {"score": 0.85}},
            {"metrics": {"score": 0.60}},
        ]
        result = compute_agent_trends(history_down, "score")
        assert result["trend_direction"] == "down"
