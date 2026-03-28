"""Tests for backend.eval.rag_comparison.output module."""

from __future__ import annotations

import json
from pathlib import Path

from backend.eval.rag_comparison.models import (
    ComparisonResults,
    ConfigCaseResult,
    ConfigResults,
)
from backend.eval.rag_comparison.output import (
    print_comparison_report,
    save_comparison_results,
)


def _make_case(correctness=0.5, relevancy=0.9, faithfulness=0.8, latency=2.0):
    return ConfigCaseResult(
        question="Q", reference_answer="A", answer="A",
        contexts=[], sources=[], latency_seconds=latency,
        answer_correctness=correctness, answer_relevancy=relevancy,
        faithfulness=faithfulness,
    )


def _make_results():
    return ComparisonResults(configs=[
        ConfigResults(config_name="vector_top5", cases=[
            _make_case(correctness=0.5, relevancy=0.9, faithfulness=0.8),
        ]),
        ConfigResults(config_name="hybrid_top5", cases=[
            _make_case(correctness=0.6, relevancy=0.95, faithfulness=0.9),
        ]),
    ])


class TestPrintComparisonReport:
    """Tests for print_comparison_report function."""

    def test_prints_empty_results(self, capsys):
        print_comparison_report(ComparisonResults())
        output = capsys.readouterr().out
        assert "No results" in output

    def test_prints_report_header(self, capsys):
        results = _make_results()
        print_comparison_report(results)
        output = capsys.readouterr().out
        assert "RAG RETRIEVAL STRATEGY COMPARISON REPORT" in output
        assert "Config" in output
        assert "Composite" in output

    def test_prints_all_configs(self, capsys):
        results = _make_results()
        print_comparison_report(results)
        output = capsys.readouterr().out
        assert "vector_top5" in output
        assert "hybrid_top5" in output

    def test_prints_winner(self, capsys):
        results = _make_results()
        print_comparison_report(results)
        output = capsys.readouterr().out
        assert "Winner:" in output
        assert "hybrid_top5" in output

    def test_prints_production_delta(self, capsys):
        results = _make_results()
        print_comparison_report(results)
        output = capsys.readouterr().out
        assert "vs Production" in output
        assert "composite delta" in output

    def test_prints_optimal_when_production_wins(self, capsys):
        results = ComparisonResults(configs=[
            ConfigResults(config_name="vector_top5", cases=[
                _make_case(correctness=0.9, relevancy=0.99, faithfulness=0.99),
            ]),
            ConfigResults(config_name="bm25_top5", cases=[
                _make_case(correctness=0.3, relevancy=0.5, faithfulness=0.4),
            ]),
        ])
        print_comparison_report(results)
        output = capsys.readouterr().out
        assert "already optimal" in output


class TestSaveComparisonResults:
    """Tests for save_comparison_results function."""

    def test_saves_json(self, tmp_path):
        results = _make_results()
        output_path = tmp_path / "results.json"
        save_comparison_results(results, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["winner"] == "hybrid_top5"
        assert len(data["configs"]) == 2

    def test_creates_parent_dirs(self, tmp_path):
        results = _make_results()
        output_path = tmp_path / "nested" / "dir" / "results.json"
        save_comparison_results(results, output_path)
        assert output_path.exists()

    def test_json_structure(self, tmp_path):
        results = _make_results()
        output_path = tmp_path / "results.json"
        save_comparison_results(results, output_path)

        with open(output_path) as f:
            data = json.load(f)
        config = data["configs"][0]
        assert "config_name" in config
        assert "avg_correctness" in config
        assert "avg_relevancy" in config
        assert "avg_faithfulness" in config
        assert "avg_composite" in config
        assert "avg_latency_seconds" in config
        assert "nan_count" in config
