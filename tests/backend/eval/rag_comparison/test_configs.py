"""Tests for backend.eval.rag_comparison.configs module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.eval.rag_comparison.configs import (
    DEFAULT_CONFIGS,
    RetrievalConfig,
    build_query_engine,
    get_configs_by_name,
)


class TestRetrievalConfig:
    """Tests for RetrievalConfig dataclass."""

    def test_defaults(self):
        config = RetrievalConfig(name="test", retriever_type="vector")
        assert config.top_k == 5
        assert config.reranker is False
        assert config.rerank_top_n == 5

    def test_str_without_reranker(self):
        config = RetrievalConfig(name="vector_top5", retriever_type="vector", top_k=5)
        assert "vector_top5" in str(config)
        assert "vector top_k=5" in str(config)

    def test_str_with_reranker(self):
        config = RetrievalConfig(
            name="vector_top10_rerank5", retriever_type="vector",
            top_k=10, reranker=True, rerank_top_n=5,
        )
        s = str(config)
        assert "rerank" in s
        assert "5" in s


class TestDefaultConfigs:
    """Tests for DEFAULT_CONFIGS list."""

    def test_has_six_configs(self):
        assert len(DEFAULT_CONFIGS) == 6

    def test_config_names(self):
        names = [c.name for c in DEFAULT_CONFIGS]
        assert "vector_top5" in names
        assert "vector_top10" in names
        assert "bm25_top5" in names
        assert "hybrid_top5" in names
        assert "vector_top10_rerank5" in names
        assert "hybrid_top10_rerank5" in names

    def test_reranker_configs(self):
        reranked = [c for c in DEFAULT_CONFIGS if c.reranker]
        assert len(reranked) == 2

    def test_retriever_types(self):
        types = {c.retriever_type for c in DEFAULT_CONFIGS}
        assert types == {"vector", "bm25", "hybrid"}


class TestGetConfigsByName:
    """Tests for get_configs_by_name function."""

    def test_returns_all_when_none(self):
        configs = get_configs_by_name(None)
        assert len(configs) == 6

    def test_filters_by_name(self):
        configs = get_configs_by_name(["vector_top5", "bm25_top5"])
        assert len(configs) == 2
        assert {c.name for c in configs} == {"vector_top5", "bm25_top5"}

    def test_raises_on_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown config names"):
            get_configs_by_name(["nonexistent"])

    def test_single_config(self):
        configs = get_configs_by_name(["hybrid_top5"])
        assert len(configs) == 1
        assert configs[0].retriever_type == "hybrid"


class TestBuildQueryEngine:
    """Tests for build_query_engine function."""

    def test_vector_retriever(self):
        config = RetrievalConfig(name="test", retriever_type="vector", top_k=5)
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        with patch("backend.eval.rag_comparison.configs.get_response_synthesizer"):
            engine = build_query_engine(config, mock_index)

        mock_index.as_retriever.assert_called_once_with(similarity_top_k=5)
        assert engine is not None

    def test_bm25_retriever(self):
        config = RetrievalConfig(name="test", retriever_type="bm25", top_k=5)
        mock_index = MagicMock()
        mock_index.storage_context.docstore.docs.values.return_value = []

        with patch("backend.eval.rag_comparison.configs.get_response_synthesizer"), \
             patch("backend.eval.rag_comparison.configs._build_bm25_retriever") as mock_bm25:
            mock_bm25.return_value = MagicMock()
            engine = build_query_engine(config, mock_index)

        mock_bm25.assert_called_once_with(mock_index, 5)
        assert engine is not None

    def test_hybrid_retriever(self):
        config = RetrievalConfig(name="test", retriever_type="hybrid", top_k=5)
        mock_index = MagicMock()

        with patch("backend.eval.rag_comparison.configs.get_response_synthesizer"), \
             patch("backend.eval.rag_comparison.configs._build_hybrid_retriever") as mock_hybrid:
            mock_hybrid.return_value = MagicMock()
            engine = build_query_engine(config, mock_index)

        mock_hybrid.assert_called_once_with(mock_index, 5)
        assert engine is not None

    def test_unknown_retriever_type_raises(self):
        config = RetrievalConfig(name="test", retriever_type="unknown")
        mock_index = MagicMock()

        with pytest.raises(ValueError, match="Unknown retriever type"):
            build_query_engine(config, mock_index)

    def test_reranker_added_when_configured(self):
        config = RetrievalConfig(
            name="test", retriever_type="vector", top_k=10,
            reranker=True, rerank_top_n=5,
        )
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = MagicMock()

        with patch("backend.eval.rag_comparison.configs.get_response_synthesizer"), \
             patch("backend.eval.rag_comparison.configs._build_reranker") as mock_reranker:
            mock_reranker.return_value = MagicMock()
            engine = build_query_engine(config, mock_index)

        mock_reranker.assert_called_once_with(5)
