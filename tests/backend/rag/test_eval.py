"""
Tests for backend.rag.eval module.

Tests the RAG evaluation models, tracking, and history functions.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Set mock mode
os.environ["MOCK_LLM"] = "1"

from backend.rag.eval.models import (
    JudgeResult,
    EvalResult,
    DocsEvalSummary,
    AccountEvalResult,
    AccountEvalSummary,
    RAGEvalSummary,
    SLO_CONTEXT_RELEVANCE,
    SLO_ANSWER_RELEVANCE,
    SLO_GROUNDEDNESS,
    SLO_RAG_TRIAD,
    SLO_DOC_RECALL,
    SLO_LATENCY_P95_MS,
)


# =============================================================================
# Model Tests
# =============================================================================

class TestJudgeResult:
    """Tests for JudgeResult model."""

    def test_judge_result_creation(self):
        """Test creating a JudgeResult."""
        result = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=1,
            needs_human_review=0,
            confidence=0.9,
            explanation="Good answer",
        )

        assert result.context_relevance == 1
        assert result.answer_relevance == 1
        assert result.groundedness == 1
        assert result.needs_human_review == 0
        assert result.confidence == 0.9

    def test_judge_result_defaults(self):
        """Test JudgeResult default values."""
        result = JudgeResult(
            context_relevance=0,
            answer_relevance=0,
            groundedness=0,
            needs_human_review=1,
        )

        assert result.confidence == 0.5
        assert result.explanation == ""

    def test_judge_result_serialization(self):
        """Test JudgeResult JSON serialization."""
        result = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=0,
            needs_human_review=0,
        )

        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["context_relevance"] == 1


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_eval_result_creation(self):
        """Test creating an EvalResult."""
        judge = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=1,
            needs_human_review=0,
        )

        result = EvalResult(
            question_id="q1",
            question="How do I import contacts?",
            target_doc_ids=["doc1", "doc2"],
            retrieved_doc_ids=["doc1", "doc3"],
            answer="You can import contacts via Settings > Import.",
            judge_result=judge,
            doc_recall=0.5,
            latency_ms=150.0,
            total_tokens=500,
        )

        assert result.question_id == "q1"
        assert result.doc_recall == 0.5
        assert result.latency_ms == 150.0

    def test_eval_result_step_timings(self):
        """Test EvalResult with step timings."""
        judge = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=1,
            needs_human_review=0,
        )

        result = EvalResult(
            question_id="q1",
            question="Test",
            target_doc_ids=[],
            retrieved_doc_ids=[],
            answer="Test answer",
            judge_result=judge,
            doc_recall=1.0,
            latency_ms=100.0,
            total_tokens=100,
            step_timings={"retrieval": 50, "generate": 40},
        )

        assert result.step_timings["retrieval"] == 50
        assert result.step_timings["generate"] == 40


class TestDocsEvalSummary:
    """Tests for DocsEvalSummary model."""

    def test_docs_eval_summary_creation(self):
        """Test creating a DocsEvalSummary."""
        summary = DocsEvalSummary(
            total_tests=10,
            context_relevance=0.85,
            answer_relevance=0.90,
            groundedness=0.80,
            rag_triad_success=0.75,
            avg_doc_recall=0.70,
            avg_latency_ms=500.0,
            p95_latency_ms=1200.0,
            total_tokens=5000,
            estimated_cost=0.05,
        )

        assert summary.total_tests == 10
        assert summary.context_relevance == 0.85
        assert summary.all_slos_passed is True
        assert summary.failed_slos == []

    def test_docs_eval_summary_with_failed_slos(self):
        """Test DocsEvalSummary with failed SLOs."""
        summary = DocsEvalSummary(
            total_tests=10,
            context_relevance=0.60,  # Below SLO
            answer_relevance=0.90,
            groundedness=0.80,
            rag_triad_success=0.50,  # Below SLO
            avg_doc_recall=0.70,
            avg_latency_ms=500.0,
            p95_latency_ms=1200.0,
            total_tokens=5000,
            estimated_cost=0.05,
            all_slos_passed=False,
            failed_slos=["context_relevance", "rag_triad"],
        )

        assert summary.all_slos_passed is False
        assert "context_relevance" in summary.failed_slos


class TestAccountEvalResult:
    """Tests for AccountEvalResult model."""

    def test_account_eval_result_creation(self):
        """Test creating an AccountEvalResult."""
        judge = JudgeResult(
            context_relevance=1,
            answer_relevance=1,
            groundedness=1,
            needs_human_review=0,
        )

        result = AccountEvalResult(
            question_id="acc1",
            company_id="ACME-MFG",
            company_name="Acme Manufacturing",
            question="What's the status of Acme Manufacturing?",
            question_type="status",
            answer="Acme Manufacturing is doing well.",
            judge_result=judge,
            privacy_leakage=0,
            leaked_company_ids=[],
            num_private_hits=5,
            latency_ms=200.0,
            total_tokens=300,
            estimated_cost=0.01,
        )

        assert result.company_id == "ACME-MFG"
        assert result.privacy_leakage == 0
        assert result.leaked_company_ids == []


class TestRAGEvalSummary:
    """Tests for RAGEvalSummary model."""

    def test_rag_eval_summary_creation(self):
        """Test creating a RAGEvalSummary."""
        docs_summary = DocsEvalSummary(
            total_tests=10,
            context_relevance=0.85,
            answer_relevance=0.90,
            groundedness=0.80,
            rag_triad_success=0.75,
            avg_doc_recall=0.70,
            avg_latency_ms=500.0,
            p95_latency_ms=1200.0,
            total_tokens=5000,
            estimated_cost=0.05,
        )

        summary = RAGEvalSummary(
            docs_eval=docs_summary,
            overall_score=0.85,
        )

        assert summary.docs_eval is not None
        assert summary.overall_score == 0.85
        assert summary.all_slos_passed is True


# =============================================================================
# SLO Constants Tests
# =============================================================================

class TestSLOConstants:
    """Tests for SLO constant values."""

    def test_slo_context_relevance(self):
        """Test SLO context relevance threshold."""
        assert SLO_CONTEXT_RELEVANCE == 0.85

    def test_slo_answer_relevance(self):
        """Test SLO answer relevance threshold."""
        assert SLO_ANSWER_RELEVANCE == 0.85

    def test_slo_groundedness(self):
        """Test SLO groundedness threshold."""
        assert SLO_GROUNDEDNESS == 0.85

    def test_slo_rag_triad(self):
        """Test SLO RAG triad threshold."""
        assert SLO_RAG_TRIAD == 0.80

    def test_slo_doc_recall(self):
        """Test SLO doc recall threshold."""
        assert SLO_DOC_RECALL == 0.80

    def test_slo_latency(self):
        """Test SLO latency threshold."""
        assert SLO_LATENCY_P95_MS == 5000


# =============================================================================
# Tracking Module Tests
# =============================================================================

class TestTrackingModule:
    """Tests for the tracking module functions."""

    def test_compare_with_previous_no_previous(self):
        """Test comparison when no previous run exists."""
        from backend.rag.eval.tracking import compare_with_previous

        current = DocsEvalSummary(
            total_tests=10,
            context_relevance=0.85,
            answer_relevance=0.90,
            groundedness=0.80,
            rag_triad_success=0.75,
            avg_doc_recall=0.70,
            avg_latency_ms=500.0,
            p95_latency_ms=1200.0,
            total_tokens=5000,
            estimated_cost=0.05,
        )

        comparison = compare_with_previous(current, None)

        assert comparison["has_previous"] is False
        assert comparison["regressions"] == []
        assert comparison["improvements"] == []

    def test_compare_with_previous_detects_regression(self):
        """Test that comparison detects regressions."""
        from backend.rag.eval.tracking import compare_with_previous

        previous = DocsEvalSummary(
            total_tests=10,
            context_relevance=0.90,
            answer_relevance=0.90,
            groundedness=0.90,
            rag_triad_success=0.85,
            avg_doc_recall=0.80,
            avg_latency_ms=400.0,
            p95_latency_ms=1000.0,
            total_tokens=5000,
            estimated_cost=0.05,
        )

        current = DocsEvalSummary(
            total_tests=10,
            context_relevance=0.70,  # Regression
            answer_relevance=0.90,
            groundedness=0.90,
            rag_triad_success=0.65,  # Regression
            avg_doc_recall=0.80,
            avg_latency_ms=400.0,
            p95_latency_ms=1000.0,
            total_tokens=5000,
            estimated_cost=0.05,
        )

        comparison = compare_with_previous(current, previous)

        assert comparison["has_previous"] is True
        assert len(comparison["regressions"]) >= 1

    def test_compare_with_previous_detects_improvement(self):
        """Test that comparison detects improvements."""
        from backend.rag.eval.tracking import compare_with_previous

        previous = DocsEvalSummary(
            total_tests=10,
            context_relevance=0.70,
            answer_relevance=0.70,
            groundedness=0.70,
            rag_triad_success=0.60,
            avg_doc_recall=0.60,
            avg_latency_ms=600.0,
            p95_latency_ms=1500.0,
            total_tokens=5000,
            estimated_cost=0.05,
        )

        current = DocsEvalSummary(
            total_tests=10,
            context_relevance=0.90,  # Improvement
            answer_relevance=0.90,  # Improvement
            groundedness=0.90,  # Improvement
            rag_triad_success=0.85,  # Improvement
            avg_doc_recall=0.80,
            avg_latency_ms=400.0,
            p95_latency_ms=900.0,  # Improvement
            total_tokens=5000,
            estimated_cost=0.05,
        )

        comparison = compare_with_previous(current, previous)

        assert comparison["has_previous"] is True
        assert len(comparison["improvements"]) >= 1


# =============================================================================
# History Module Tests
# =============================================================================

class TestHistoryModule:
    """Tests for the history module functions."""

    def test_compute_trends_insufficient_data(self):
        """Test trend computation with insufficient data."""
        from backend.rag.eval.history import compute_trends

        history = [{"metrics": {"rag_triad": 0.8}}]
        result = compute_trends(history, "rag_triad")

        assert result["has_trend"] is False

    def test_compute_trends_with_data(self):
        """Test trend computation with sufficient data."""
        from backend.rag.eval.history import compute_trends

        history = [
            {"metrics": {"rag_triad": 0.70}},
            {"metrics": {"rag_triad": 0.75}},
            {"metrics": {"rag_triad": 0.80}},
        ]

        result = compute_trends(history, "rag_triad")

        assert result["has_trend"] is True
        assert result["min"] == 0.70
        assert result["max"] == 0.80
        assert result["current"] == 0.80

    def test_compute_trends_direction(self):
        """Test trend direction computation."""
        from backend.rag.eval.history import compute_trends

        # Upward trend
        history_up = [
            {"metrics": {"score": 0.60}},
            {"metrics": {"score": 0.80}},
        ]
        result = compute_trends(history_up, "score")
        assert result["trend_direction"] == "up"

        # Downward trend
        history_down = [
            {"metrics": {"score": 0.80}},
            {"metrics": {"score": 0.60}},
        ]
        result = compute_trends(history_down, "score")
        assert result["trend_direction"] == "down"

    def test_detect_degradation_empty(self):
        """Test degradation detection with no history."""
        from backend.rag.eval.history import detect_degradation

        result = detect_degradation([])
        assert result == []

    def test_detect_degradation_finds_issues(self):
        """Test that degradation detection finds degrading metrics."""
        from backend.rag.eval.history import detect_degradation

        history = [
            {"metrics": {"rag_triad": 0.90, "context_relevance": 0.90, "p95_latency_ms": 1000}},
            {"metrics": {"rag_triad": 0.70, "context_relevance": 0.60, "p95_latency_ms": 1200}},
        ]

        result = detect_degradation(history, threshold=0.05)

        # Should detect at least one degradation
        assert len(result) >= 1
