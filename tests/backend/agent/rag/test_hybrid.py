"""
Tests for hybrid search functionality.

Covers hybrid search configuration and integration.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch


class TestHybridConfig:
    """Tests for hybrid search configuration."""

    def test_hybrid_config_exists(self):
        """Test that hybrid config values are present."""
        from backend.agent.fetch.rag.config import (
            SPARSE_EMBEDDING_MODEL,
            SEARCH_TOP_K,
        )

        assert SPARSE_EMBEDDING_MODEL == "Qdrant/bm25"
        assert SEARCH_TOP_K > 0

    def test_hybrid_config_exported(self):
        """Test hybrid config is in __all__."""
        from backend.agent.fetch.rag import config

        assert "SPARSE_EMBEDDING_MODEL" in config.__all__
        assert "SEARCH_TOP_K" in config.__all__


class TestHybridIngestion:
    """Tests for hybrid ingestion."""

    def test_hybrid_config_imported_in_ingest(self):
        """Test that ingest module imports hybrid config."""
        from backend.agent.fetch.rag import ingest

        # Verify hybrid config is available in the module
        assert hasattr(ingest, "SPARSE_EMBEDDING_MODEL")


class TestHybridRetrieval:
    """Tests for hybrid retrieval in search_entity_context."""

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

        # Mock the postprocessor module for SentenceTransformerRerank
        mock_core_postprocessor = MagicMock()
        mock_core_postprocessor.SentenceTransformerRerank = MagicMock()

        mock_hf = MagicMock()
        mock_hf.HuggingFaceEmbedding = mock_embed_cls

        mock_qdrant_vs = MagicMock()
        mock_qdrant_vs.QdrantVectorStore = mock_vector_store_cls

        return {
            "llama_index.core": mock_core,
            "llama_index.core.postprocessor": mock_core_postprocessor,
            "llama_index.embeddings.huggingface": mock_hf,
            "llama_index.vector_stores.qdrant": mock_qdrant_vs,
            "mock_vector_store_cls": mock_vector_store_cls,
            "mock_index": mock_index,
        }

    @pytest.mark.no_mock_llm
    def test_retriever_uses_hybrid_mode(self):
        """Test retriever sets vector_store_query_mode='hybrid'."""
        mock_nodes = [MagicMock() for _ in range(3)]
        for i, node in enumerate(mock_nodes):
            node.text = f"Content {i}"
            node.metadata = {"type": "note", "source_id": f"note_{i}"}

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_nodes

        mocks = self._setup_llama_mocks(mock_retriever)
        llama_mocks = {
            "llama_index.core": mocks["llama_index.core"],
            "llama_index.core.postprocessor": mocks["llama_index.core.postprocessor"],
            "llama_index.embeddings.huggingface": mocks["llama_index.embeddings.huggingface"],
            "llama_index.vector_stores.qdrant": mocks["llama_index.vector_stores.qdrant"],
        }
        mock_index = mocks["mock_index"]

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with (
                patch.object(search, "get_qdrant_client") as mock_client,
                patch.object(search, "SEARCH_TOP_K", 15),
            ):
                mock_client.return_value = MagicMock()
                search.search_entity_context("test query", {"company_id": "COMP001"})

        # Verify retriever was called with hybrid mode
        mock_index.as_retriever.assert_called_once()
        call_kwargs = mock_index.as_retriever.call_args[1]
        assert call_kwargs.get("vector_store_query_mode") == "hybrid"
        assert call_kwargs.get("sparse_top_k") == 15

    @pytest.mark.no_mock_llm
    def test_vector_store_created_with_hybrid_params(self):
        """Test QdrantVectorStore is created with hybrid params."""
        mock_nodes = []
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_nodes

        mocks = self._setup_llama_mocks(mock_retriever)
        llama_mocks = {
            "llama_index.core": mocks["llama_index.core"],
            "llama_index.core.postprocessor": mocks["llama_index.core.postprocessor"],
            "llama_index.embeddings.huggingface": mocks["llama_index.embeddings.huggingface"],
            "llama_index.vector_stores.qdrant": mocks["llama_index.vector_stores.qdrant"],
        }
        mock_vector_store_cls = mocks["mock_vector_store_cls"]

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                search.search_entity_context("test query", {"company_id": "COMP001"})

        # Verify QdrantVectorStore was called with hybrid params
        mock_vector_store_cls.assert_called_once()
        call_kwargs = mock_vector_store_cls.call_args[1]
        assert call_kwargs.get("enable_hybrid") is True
        assert call_kwargs.get("fastembed_sparse_model") == "Qdrant/bm25"
