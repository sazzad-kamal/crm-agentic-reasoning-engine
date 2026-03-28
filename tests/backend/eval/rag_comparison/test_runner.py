"""Tests for backend.eval.rag_comparison.runner module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.eval.rag_comparison.runner import (
    _load_questions,
    _run_single_question,
    run_rag_comparison,
)


class TestLoadQuestions:
    """Tests for _load_questions function."""

    def test_loads_all_questions(self):
        questions = _load_questions()
        assert len(questions) == 20
        assert "question" in questions[0]
        assert "reference_answer" in questions[0]

    def test_limit(self):
        questions = _load_questions(limit=3)
        assert len(questions) == 3

    def test_limit_none_returns_all(self):
        questions = _load_questions(limit=None)
        assert len(questions) == 20

    def test_limit_exceeds_total(self):
        questions = _load_questions(limit=100)
        assert len(questions) == 20


class TestRunSingleQuestion:
    """Tests for _run_single_question function."""

    def test_returns_config_case_result(self):
        mock_node = MagicMock()
        mock_node.text = "Some context"
        mock_node.metadata = {"source": "doc.pdf"}

        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "The answer"
        mock_response.source_nodes = [mock_node]

        mock_engine = MagicMock()
        mock_engine.query.return_value = mock_response

        ragas_scores = {
            "answer_correctness": 0.8,
            "answer_relevancy": 0.9,
            "faithfulness": 0.95,
            "nan_metrics": [],
        }

        with patch("backend.eval.rag_comparison.runner.evaluate_single", return_value=ragas_scores):
            result = _run_single_question(mock_engine, "What is Act!?", "A CRM tool")

        assert result.question == "What is Act!?"
        assert result.answer == "The answer"
        assert result.contexts == ["Some context"]
        assert result.sources == ["doc.pdf"]
        assert result.answer_correctness == 0.8
        assert result.answer_relevancy == 0.9
        assert result.faithfulness == 0.95
        assert result.latency_seconds >= 0


class TestRunRagComparison:
    """Tests for run_rag_comparison function."""

    def test_runs_specified_configs(self):
        mock_node = MagicMock()
        mock_node.text = "context"
        mock_node.metadata = {"source": "doc.pdf"}

        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "answer"
        mock_response.source_nodes = [mock_node]

        mock_engine = MagicMock()
        mock_engine.query.return_value = mock_response

        ragas_scores = {
            "answer_correctness": 0.8,
            "answer_relevancy": 0.9,
            "faithfulness": 0.95,
            "nan_metrics": [],
        }

        with patch("backend.eval.rag_comparison.runner.get_index") as mock_index, \
             patch("backend.eval.rag_comparison.runner.build_query_engine", return_value=mock_engine), \
             patch("backend.eval.rag_comparison.runner.evaluate_single", return_value=ragas_scores):
            results = run_rag_comparison(config_names=["vector_top5"], limit=2)

        assert len(results.configs) == 1
        assert results.configs[0].config_name == "vector_top5"
        assert len(results.configs[0].cases) == 2

    def test_handles_question_failure(self):
        mock_engine = MagicMock()
        mock_engine.query.side_effect = Exception("API error")

        with patch("backend.eval.rag_comparison.runner.get_index"), \
             patch("backend.eval.rag_comparison.runner.build_query_engine", return_value=mock_engine):
            results = run_rag_comparison(config_names=["vector_top5"], limit=1)

        assert len(results.configs[0].cases) == 1
        assert "ERROR" in results.configs[0].cases[0].answer
        assert len(results.configs[0].cases[0].nan_metrics) == 3
