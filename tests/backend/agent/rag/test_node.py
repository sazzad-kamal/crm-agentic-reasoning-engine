"""Tests for RAG node."""

from unittest.mock import MagicMock, patch

import pytest

from backend.agent.rag.node import rag_node


class TestRagNode:
    """Tests for the RAG LangGraph node."""

    def test_rag_node_returns_results_on_success(self):
        """Should return RAG results in sql_results format."""
        mock_result = MagicMock()
        mock_result.answer = "To import contacts, go to File > Import."
        mock_result.sources = [
            {"text": "Import contacts from File menu...", "source": "quick-start.pdf", "score": 0.9},
            {"text": "The import wizard guides you...", "source": "guide.pdf", "score": 0.8},
        ]
        mock_result.confidence = 0.85

        with patch("backend.agent.rag.node.retrieve_and_answer", return_value=mock_result):
            state = {"question": "How do I import contacts?"}
            result = rag_node(state)

            assert "sql_results" in result
            assert result["sql_results"]["rag_answer"] == "To import contacts, go to File > Import."
            assert result["sql_results"]["rag_confidence"] == 0.85
            assert len(result["sql_results"]["rag_sources"]) == 2
            assert result["sql_results"]["_source"] == "documentation"

    def test_rag_node_handles_empty_question(self):
        """Should handle empty question gracefully."""
        mock_result = MagicMock()
        mock_result.answer = ""
        mock_result.sources = []
        mock_result.confidence = 0.0

        with patch("backend.agent.rag.node.retrieve_and_answer", return_value=mock_result):
            state = {"question": ""}
            result = rag_node(state)

            assert "sql_results" in result
            assert result["sql_results"]["rag_confidence"] == 0.0

    def test_rag_node_handles_exception(self):
        """Should return error state on exception."""
        with patch("backend.agent.rag.node.retrieve_and_answer", side_effect=Exception("Index not found")):
            state = {"question": "How do I create groups?"}
            result = rag_node(state)

            assert "sql_results" in result
            assert result["sql_results"]["rag_answer"] is None
            assert "error" in result["sql_results"]
            assert "Index not found" in result["sql_results"]["error"]

    def test_rag_node_truncates_sources(self):
        """Should truncate source excerpts to reasonable length."""
        mock_result = MagicMock()
        mock_result.answer = "Here's how to do it."
        mock_result.sources = [
            {"text": "A" * 500, "source": "doc.pdf", "score": 0.9},
        ]
        mock_result.confidence = 0.9

        with patch("backend.agent.rag.node.retrieve_and_answer", return_value=mock_result):
            state = {"question": "How do I do something?"}
            result = rag_node(state)

            # Should truncate to ~200 chars + "..."
            excerpt = result["sql_results"]["rag_sources"][0]["excerpt"]
            assert len(excerpt) < 250
            assert excerpt.endswith("...")

    def test_rag_node_assigns_evidence_ids(self):
        """Should assign D1, D2, D3 evidence IDs to sources."""
        mock_result = MagicMock()
        mock_result.answer = "Answer"
        mock_result.sources = [
            {"text": "First", "source": "a.pdf", "score": 0.9},
            {"text": "Second", "source": "b.pdf", "score": 0.8},
            {"text": "Third", "source": "c.pdf", "score": 0.7},
        ]
        mock_result.confidence = 0.8

        with patch("backend.agent.rag.node.retrieve_and_answer", return_value=mock_result):
            state = {"question": "Test"}
            result = rag_node(state)

            sources = result["sql_results"]["rag_sources"]
            assert sources[0]["id"] == "D1"
            assert sources[1]["id"] == "D2"
            assert sources[2]["id"] == "D3"
