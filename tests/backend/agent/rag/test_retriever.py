"""Tests for RAG retriever."""

from unittest.mock import MagicMock, patch

import pytest

from backend.agent.rag.retriever import RAGResult, retrieve_and_answer, search_docs


class TestRAGResult:
    """Tests for RAGResult dataclass."""

    def test_rag_result_has_required_fields(self):
        """Should have answer, sources, and confidence fields."""
        result = RAGResult(
            answer="Test answer",
            sources=[{"text": "source", "source": "doc.pdf", "score": 0.9}],
            confidence=0.85,
        )
        assert result.answer == "Test answer"
        assert len(result.sources) == 1
        assert result.confidence == 0.85


class TestRetrieveAndAnswer:
    """Tests for retrieve_and_answer function."""

    def test_returns_rag_result(self):
        """Should return RAGResult with answer and sources."""
        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "This is the answer."

        mock_node = MagicMock()
        mock_node.text = "Source text from document"
        mock_node.metadata = {"source": "guide.pdf"}
        mock_node.score = 0.9
        mock_response.source_nodes = [mock_node]

        mock_query_engine = MagicMock()
        mock_query_engine.query.return_value = mock_response

        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine

        with patch("backend.agent.rag.retriever.get_index", return_value=mock_index):
            result = retrieve_and_answer("How do I import contacts?")

            assert isinstance(result, RAGResult)
            assert result.answer == "This is the answer."
            assert len(result.sources) == 1
            assert result.sources[0]["source"] == "guide.pdf"

    def test_handles_exception_gracefully(self):
        """Should return fallback result on exception."""
        with patch("backend.agent.rag.retriever.get_index", side_effect=Exception("Index error")):
            result = retrieve_and_answer("Test question")

            assert isinstance(result, RAGResult)
            assert "couldn't find" in result.answer.lower()
            assert result.sources == []
            assert result.confidence == 0.0

    def test_truncates_long_source_text(self):
        """Should truncate source text longer than 500 chars."""
        mock_response = MagicMock()
        mock_response.__str__ = lambda self: "Answer"

        mock_node = MagicMock()
        mock_node.text = "A" * 1000
        mock_node.metadata = {"source": "doc.pdf"}
        mock_node.score = 0.9
        mock_response.source_nodes = [mock_node]

        mock_query_engine = MagicMock()
        mock_query_engine.query.return_value = mock_response

        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine

        with patch("backend.agent.rag.retriever.get_index", return_value=mock_index):
            result = retrieve_and_answer("Test")

            # Should be truncated to 500 + "..."
            assert len(result.sources[0]["text"]) == 503


class TestSearchDocs:
    """Tests for search_docs function."""

    def test_returns_list_of_results(self):
        """Should return list of matching documents."""
        mock_node = MagicMock()
        mock_node.text = "Document content"
        mock_node.metadata = {"source": "doc.pdf", "page": 5}
        mock_node.score = 0.85

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        with patch("backend.agent.rag.retriever.get_index", return_value=mock_index):
            results = search_docs("import contacts")

            assert len(results) == 1
            assert results[0]["text"] == "Document content"
            assert results[0]["source"] == "doc.pdf"
            assert results[0]["score"] == 0.85

    def test_handles_exception(self):
        """Should return empty list on exception."""
        with patch("backend.agent.rag.retriever.get_index", side_effect=Exception("Error")):
            results = search_docs("test query")
            assert results == []
