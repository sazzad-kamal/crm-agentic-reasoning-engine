"""
Tests for reranker module.

Covers rerank_nodes function with mocked dependencies.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestRerankerConfig:
    """Tests for reranker configuration."""

    def test_reranker_config_exists(self):
        """Test that reranker config values are present."""
        from backend.agent.rag.config import (
            RERANKER_ENABLED,
            RERANKER_MODEL,
            RERANKER_TOP_K,
            RETRIEVAL_TOP_K,
        )

        assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert RERANKER_TOP_K == 5
        assert RETRIEVAL_TOP_K == 30  # Over-retrieve before reranking
        assert isinstance(RERANKER_ENABLED, bool)


class TestRerankNodes:
    """Tests for rerank_nodes function."""

    def test_rerank_empty_list(self):
        """Test reranking handles empty input."""
        from backend.agent.rag.reranker import rerank_nodes

        result = rerank_nodes([], "any query")
        assert result == []

    def test_rerank_under_limit_passthrough(self):
        """Test nodes under top_k are returned as-is without reranking."""
        # Create mock nodes
        mock_nodes = [MagicMock() for _ in range(3)]
        for i, node in enumerate(mock_nodes):
            node.text = f"Node {i} content"

        from backend.agent.rag.reranker import rerank_nodes

        # With top_k=5 and only 3 nodes, should return as-is
        result = rerank_nodes(mock_nodes, "test query", top_k=5)
        assert result == mock_nodes

    def test_rerank_returns_top_k(self):
        """Test reranking returns correct number of nodes."""
        # Create mock nodes
        mock_nodes = [MagicMock() for _ in range(10)]
        for i, node in enumerate(mock_nodes):
            node.text = f"Node {i} content"
            node.score = 1.0 - (i * 0.1)  # Decreasing scores

        # Mock the reranker postprocessor
        mock_reranker = MagicMock()
        mock_reranker.postprocess_nodes.return_value = mock_nodes[:3]  # Return top 3

        with patch("backend.agent.rag.reranker._get_reranker", return_value=mock_reranker):
            from backend.agent.rag.reranker import rerank_nodes

            result = rerank_nodes(mock_nodes, "test query", top_k=3)

        assert len(result) == 3
        mock_reranker.postprocess_nodes.assert_called_once()

    def test_rerank_uses_config_default(self):
        """Test reranking uses config RERANKER_TOP_K when top_k not specified."""
        mock_nodes = [MagicMock() for _ in range(10)]
        for i, node in enumerate(mock_nodes):
            node.text = f"Node {i} content"

        mock_reranker = MagicMock()
        mock_reranker.postprocess_nodes.return_value = mock_nodes[:5]

        with patch("backend.agent.rag.reranker._get_reranker", return_value=mock_reranker):
            from backend.agent.rag.reranker import rerank_nodes

            # Don't specify top_k, should use default from config (5)
            result = rerank_nodes(mock_nodes, "test query")

        assert len(result) == 5

    def test_rerank_passes_query_bundle(self):
        """Test reranking passes correct query to postprocessor."""
        mock_nodes = [MagicMock() for _ in range(10)]
        for node in mock_nodes:
            node.text = "content"

        mock_reranker = MagicMock()
        mock_reranker.postprocess_nodes.return_value = mock_nodes[:5]

        with patch("backend.agent.rag.reranker._get_reranker", return_value=mock_reranker):
            from backend.agent.rag.reranker import rerank_nodes

            rerank_nodes(mock_nodes, "my specific query", top_k=5)

        # Verify query was passed correctly
        call_args = mock_reranker.postprocess_nodes.call_args
        assert call_args is not None
        query_bundle = call_args[0][1]
        assert query_bundle.query_str == "my specific query"


class TestGetReranker:
    """Tests for _get_reranker singleton."""

    def test_get_reranker_caching(self):
        """Test that _get_reranker uses singleton pattern (thread-safe global)."""
        mock_rerank_cls = MagicMock()
        mock_instance = MagicMock()
        mock_rerank_cls.return_value = mock_instance

        mock_postprocessor = MagicMock()
        mock_postprocessor.SentenceTransformerRerank = mock_rerank_cls

        with patch.dict(
            "sys.modules",
            {"llama_index.core.postprocessor": mock_postprocessor},
        ):
            # Reset the global singleton
            from backend.agent.rag import reranker

            reranker._reranker = None

            # First call should create instance
            result1 = reranker._get_reranker()
            # Second call should return cached instance
            result2 = reranker._get_reranker()

            assert result1 is result2
            # Constructor should only be called once due to singleton
            assert mock_rerank_cls.call_count == 1

            # Clean up
            reranker._reranker = None


class TestToolAccountRagWithReranker:
    """Tests for tool_account_rag with reranker integration."""

    def _setup_llama_mocks(self, mock_retriever):
        """Setup common llama_index mocks."""
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        mock_index_cls = MagicMock()
        mock_index_cls.from_vector_store.return_value = mock_index

        mock_settings = MagicMock()
        mock_vector_store_cls = MagicMock()
        mock_embed_cls = MagicMock()

        mock_core = MagicMock()
        mock_core.Settings = mock_settings
        mock_core.VectorStoreIndex = mock_index_cls

        mock_hf = MagicMock()
        mock_hf.HuggingFaceEmbedding = mock_embed_cls

        mock_qdrant_vs = MagicMock()
        mock_qdrant_vs.QdrantVectorStore = mock_vector_store_cls

        return {
            "llama_index.core": mock_core,
            "llama_index.embeddings.huggingface": mock_hf,
            "llama_index.vector_stores.qdrant": mock_qdrant_vs,
        }

    def test_tool_account_rag_calls_reranker_when_enabled(self):
        """Test that tool_account_rag calls reranker when enabled."""
        import sys

        # Create mock nodes (more than RERANKER_TOP_K to trigger reranking)
        mock_nodes = [MagicMock() for _ in range(10)]
        for i, node in enumerate(mock_nodes):
            node.text = f"Content {i}"
            node.metadata = {"type": "note", "source_id": f"note_{i}"}

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_nodes

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        # Mock rerank_nodes to return top 5
        reranked_nodes = mock_nodes[:5]

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.rag.tools" in sys.modules:
                del sys.modules["backend.agent.rag.tools"]

            from backend.agent.rag import tools

            with (
                patch.object(tools, "get_qdrant_client") as mock_client,
                patch.object(tools, "RERANKER_ENABLED", True),
                patch.object(tools, "RERANKER_TOP_K", 5),
                patch.object(tools, "RETRIEVAL_TOP_K", 15),
                patch("backend.agent.rag.reranker.rerank_nodes", return_value=reranked_nodes) as mock_rerank,
            ):
                mock_client.return_value = MagicMock()
                context, sources = tools.tool_account_rag("test query", "COMP001")

        # Should have 5 sources (after reranking)
        assert len(sources) == 5

    def test_tool_account_rag_skips_reranker_when_disabled(self):
        """Test that tool_account_rag skips reranker when disabled."""
        import sys

        mock_nodes = [MagicMock() for _ in range(3)]
        for i, node in enumerate(mock_nodes):
            node.text = f"Content {i}"
            node.metadata = {"type": "note", "source_id": f"note_{i}"}

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_nodes

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.rag.tools" in sys.modules:
                del sys.modules["backend.agent.rag.tools"]

            from backend.agent.rag import tools

            with (
                patch.object(tools, "get_qdrant_client") as mock_client,
                patch.object(tools, "RERANKER_ENABLED", False),
            ):
                mock_client.return_value = MagicMock()
                context, sources = tools.tool_account_rag("test query", "COMP001")

        # Should have all 3 sources (no reranking)
        assert len(sources) == 3
