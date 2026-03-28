"""Tests for backend.eval.rag_comparison.models module."""

from __future__ import annotations

import pytest

from backend.eval.rag_comparison.models import (
    ComparisonResults,
    ConfigCaseResult,
    ConfigResults,
)


class TestConfigCaseResult:
    """Tests for ConfigCaseResult dataclass."""

    def test_basic_creation(self):
        case = ConfigCaseResult(
            question="What is Act!?",
            reference_answer="A CRM tool",
            answer="Act! is a CRM",
            contexts=["ctx1"],
            sources=["doc.pdf"],
            latency_seconds=1.5,
        )
        assert case.question == "What is Act!?"
        assert case.latency_seconds == 1.5
        assert case.answer_correctness == 0.0
        assert case.nan_metrics == []

    def test_composite_score(self):
        case = ConfigCaseResult(
            question="Q",
            reference_answer="A",
            answer="A",
            contexts=[],
            sources=[],
            latency_seconds=1.0,
            answer_correctness=0.5,
            answer_relevancy=0.9,
            faithfulness=0.8,
        )
        # 0.4 * 0.9 + 0.4 * 0.8 + 0.2 * 0.5 = 0.36 + 0.32 + 0.10 = 0.78
        assert abs(case.composite_score - 0.78) < 1e-9

    def test_composite_score_all_zeros(self):
        case = ConfigCaseResult(
            question="Q", reference_answer="A", answer="A",
            contexts=[], sources=[], latency_seconds=0.0,
        )
        assert case.composite_score == 0.0

    def test_composite_score_perfect(self):
        case = ConfigCaseResult(
            question="Q", reference_answer="A", answer="A",
            contexts=[], sources=[], latency_seconds=0.0,
            answer_correctness=1.0, answer_relevancy=1.0, faithfulness=1.0,
        )
        assert abs(case.composite_score - 1.0) < 1e-9


class TestConfigResults:
    """Tests for ConfigResults dataclass."""

    def _make_case(self, correctness=0.5, relevancy=0.9, faithfulness=0.8, latency=2.0):
        return ConfigCaseResult(
            question="Q", reference_answer="A", answer="A",
            contexts=[], sources=[], latency_seconds=latency,
            answer_correctness=correctness, answer_relevancy=relevancy,
            faithfulness=faithfulness,
        )

    def test_empty_cases(self):
        cr = ConfigResults(config_name="test")
        assert cr.avg_correctness == 0.0
        assert cr.avg_relevancy == 0.0
        assert cr.avg_faithfulness == 0.0
        assert cr.avg_composite == 0.0
        assert cr.avg_latency == 0.0
        assert cr.p90_latency == 0.0
        assert cr.total_nan_count == 0

    def test_averages(self):
        cr = ConfigResults(config_name="test", cases=[
            self._make_case(correctness=0.4, relevancy=0.8, faithfulness=0.6),
            self._make_case(correctness=0.6, relevancy=1.0, faithfulness=1.0),
        ])
        assert abs(cr.avg_correctness - 0.5) < 1e-9
        assert abs(cr.avg_relevancy - 0.9) < 1e-9
        assert abs(cr.avg_faithfulness - 0.8) < 1e-9

    def test_p90_latency(self):
        cr = ConfigResults(config_name="test", cases=[
            self._make_case(latency=1.0),
            self._make_case(latency=2.0),
            self._make_case(latency=3.0),
            self._make_case(latency=4.0),
            self._make_case(latency=5.0),
            self._make_case(latency=6.0),
            self._make_case(latency=7.0),
            self._make_case(latency=8.0),
            self._make_case(latency=9.0),
            self._make_case(latency=10.0),
        ])
        assert cr.p90_latency == 10.0

    def test_total_nan_count(self):
        case1 = self._make_case()
        case1.nan_metrics = ["faithfulness"]
        case2 = self._make_case()
        case2.nan_metrics = ["answer_correctness", "answer_relevancy"]
        cr = ConfigResults(config_name="test", cases=[case1, case2])
        assert cr.total_nan_count == 3

    def test_to_dict(self):
        cr = ConfigResults(config_name="vector_top5", cases=[
            self._make_case(correctness=0.5, relevancy=0.9, faithfulness=0.8, latency=3.0),
        ])
        d = cr.to_dict()
        assert d["config_name"] == "vector_top5"
        assert d["num_questions"] == 1
        assert d["avg_correctness"] == 0.5
        assert d["avg_latency_seconds"] == 3.0
        assert d["nan_count"] == 0


class TestComparisonResults:
    """Tests for ComparisonResults dataclass."""

    def _make_config_results(self, name, composite):
        case = ConfigCaseResult(
            question="Q", reference_answer="A", answer="A",
            contexts=[], sources=[], latency_seconds=1.0,
            answer_correctness=composite, answer_relevancy=composite,
            faithfulness=composite,
        )
        return ConfigResults(config_name=name, cases=[case])

    def test_empty_results(self):
        cr = ComparisonResults()
        assert cr.winner is None
        assert cr.production_config is None

    def test_winner_selection(self):
        cr = ComparisonResults(configs=[
            self._make_config_results("low", 0.5),
            self._make_config_results("high", 0.9),
            self._make_config_results("mid", 0.7),
        ])
        assert cr.winner.config_name == "high"

    def test_production_config_found(self):
        cr = ComparisonResults(configs=[
            self._make_config_results("vector_top5", 0.8),
            self._make_config_results("bm25_top5", 0.6),
        ])
        assert cr.production_config.config_name == "vector_top5"

    def test_production_config_not_found(self):
        cr = ComparisonResults(configs=[
            self._make_config_results("bm25_top5", 0.6),
        ])
        assert cr.production_config is None

    def test_to_dict(self):
        cr = ComparisonResults(configs=[
            self._make_config_results("vector_top5", 0.8),
            self._make_config_results("hybrid_top5", 0.9),
        ])
        d = cr.to_dict()
        assert d["winner"] == "hybrid_top5"
        assert len(d["configs"]) == 2
        assert d["winner_composite"] is not None
