"""
Tests for ranking utilities (RRF and reranking).

Tests RankingMixin, RRF merging, and cross-encoder reranking.

Run with:
    pytest tests/backend/rag/test_ranking.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.rag.models import DocumentChunk
from backend.rag.retrieval.ranking import RankingMixin


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def ranking_mixin():
    """Create a RankingMixin instance for testing."""
    return RankingMixin()


@pytest.fixture
def sample_chunks():
    """Create sample document chunks."""
    return [
        DocumentChunk(
            chunk_id=f"doc{i}::0",
            doc_id=f"doc{i}",
            title=f"Document {i}",
            text=f"Content for document {i}",
            metadata={},
        )
        for i in range(5)
    ]


# =============================================================================
# Test: RRF Merging
# =============================================================================

class TestRRFMerge:
    """Tests for RRF (Reciprocal Rank Fusion) merging."""

    def test_rrf_merge_basic(self, ranking_mixin):
        """Test basic RRF merging of two result lists."""
        dense_results = [(0, 0.9), (1, 0.8), (2, 0.7)]
        bm25_results = [(1, 10.0), (0, 8.0), (3, 5.0)]

        merged = ranking_mixin.rrf_merge(dense_results, bm25_results)

        assert len(merged) > 0
        assert all(isinstance(item, tuple) for item in merged)
        assert all(len(item) == 2 for item in merged)

    def test_rrf_merge_sorted(self, ranking_mixin):
        """Test that RRF results are sorted by score."""
        dense_results = [(0, 0.9), (1, 0.5)]
        bm25_results = [(1, 10.0), (0, 5.0)]

        merged = ranking_mixin.rrf_merge(dense_results, bm25_results)

        scores = [score for _, score in merged]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_merge_combines_rankings(self, ranking_mixin):
        """Test that items appearing in both lists get higher scores."""
        # Item 0 appears in both lists
        dense_results = [(0, 0.9), (1, 0.8)]
        bm25_results = [(0, 10.0), (2, 8.0)]

        merged = ranking_mixin.rrf_merge(dense_results, bm25_results)

        # Item 0 should have highest RRF score
        top_item, top_score = merged[0]
        assert top_item == 0

    def test_rrf_empty_dense(self, ranking_mixin):
        """Test RRF with empty dense results."""
        dense_results = []
        bm25_results = [(0, 10.0), (1, 8.0)]

        merged = ranking_mixin.rrf_merge(dense_results, bm25_results)

        # Should still return BM25 results
        assert len(merged) > 0

    def test_rrf_empty_bm25(self, ranking_mixin):
        """Test RRF with empty BM25 results."""
        dense_results = [(0, 0.9), (1, 0.8)]
        bm25_results = []

        merged = ranking_mixin.rrf_merge(dense_results, bm25_results)

        # Should still return dense results
        assert len(merged) > 0

    def test_rrf_both_empty(self, ranking_mixin):
        """Test RRF with both lists empty."""
        merged = ranking_mixin.rrf_merge([], [])

        assert merged == []

    def test_rrf_custom_k(self, ranking_mixin):
        """Test RRF with custom k parameter."""
        dense_results = [(0, 0.9), (1, 0.8)]
        bm25_results = [(0, 10.0), (1, 8.0)]

        merged_k60 = ranking_mixin.rrf_merge(dense_results, bm25_results, k=60)
        merged_k10 = ranking_mixin.rrf_merge(dense_results, bm25_results, k=10)

        # Different k values should give different scores
        assert merged_k60 != merged_k10


# =============================================================================
# Test: Reranking
# =============================================================================

class TestRerank:
    """Tests for cross-encoder reranking."""

    @patch('backend.rag.retrieval.ranking.CrossEncoder')
    def test_rerank_basic(self, mock_cross_encoder, ranking_mixin, sample_chunks):
        """Test basic reranking functionality."""
        # Mock the cross-encoder
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.5, 0.8, 0.6]
        mock_cross_encoder.return_value = mock_model
        ranking_mixin._reranker = mock_model

        query = "test query"
        reranked = ranking_mixin.rerank(query, sample_chunks, top_n=3)

        assert len(reranked) == 3
        assert all(isinstance(item, tuple) for item in reranked)

    @patch('backend.rag.retrieval.ranking.CrossEncoder')
    def test_rerank_sorted_by_score(self, mock_cross_encoder, ranking_mixin, sample_chunks):
        """Test that reranked results are sorted by score."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.3, 0.7, 0.1]
        mock_cross_encoder.return_value = mock_model
        ranking_mixin._reranker = mock_model

        reranked = ranking_mixin.rerank("query", sample_chunks, top_n=5)

        scores = [score for _, score in reranked]
        assert scores == sorted(scores, reverse=True)

    @patch('backend.rag.retrieval.ranking.CrossEncoder')
    def test_rerank_respects_top_n(self, mock_cross_encoder, ranking_mixin, sample_chunks):
        """Test that rerank respects top_n parameter."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        mock_cross_encoder.return_value = mock_model
        ranking_mixin._reranker = mock_model

        reranked = ranking_mixin.rerank("query", sample_chunks, top_n=2)

        assert len(reranked) == 2

    @patch('backend.rag.retrieval.ranking.CrossEncoder')
    def test_rerank_empty_candidates(self, mock_cross_encoder, ranking_mixin):
        """Test reranking with empty candidate list."""
        reranked = ranking_mixin.rerank("query", [], top_n=5)

        assert reranked == []

    @patch('backend.rag.retrieval.ranking.CrossEncoder')
    def test_rerank_creates_query_doc_pairs(self, mock_cross_encoder, ranking_mixin, sample_chunks):
        """Test that rerank creates correct query-document pairs."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]
        mock_cross_encoder.return_value = mock_model
        ranking_mixin._reranker = mock_model

        query = "test query"
        ranking_mixin.rerank(query, sample_chunks[:3], top_n=3)

        # Check that predict was called with pairs
        mock_model.predict.assert_called_once()
        pairs = mock_model.predict.call_args[0][0]
        assert all(pair[0] == query for pair in pairs)

    @patch('backend.rag.retrieval.ranking.CrossEncoder')
    def test_rerank_top_n_larger_than_candidates(self, mock_cross_encoder, ranking_mixin, sample_chunks):
        """Test rerank when top_n is larger than number of candidates."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8]
        mock_cross_encoder.return_value = mock_model
        ranking_mixin._reranker = mock_model

        reranked = ranking_mixin.rerank("query", sample_chunks[:2], top_n=10)

        # Should return all candidates
        assert len(reranked) == 2


# =============================================================================
# Test: Lazy Loading
# =============================================================================

class TestLazyLoading:
    """Tests for lazy loading of models."""

    @patch('backend.rag.retrieval.preload._reranker_model', None)
    @patch('backend.rag.retrieval.preload.get_reranker_model')
    def test_reranker_lazy_loaded(self, mock_get_reranker):
        """Test that reranker is lazy loaded via preload module."""
        mock_model = MagicMock()
        mock_get_reranker.return_value = mock_model
        mixin = RankingMixin()

        # Should not be loaded initially
        assert mixin._reranker is None

        # Access property to trigger loading
        _ = mixin.reranker

        # Should be loaded now via get_reranker_model
        mock_get_reranker.assert_called_once()

    @patch('backend.rag.retrieval.preload._reranker_model', None)
    @patch('backend.rag.retrieval.preload.get_reranker_model')
    def test_reranker_loaded_once(self, mock_get_reranker):
        """Test that reranker is only loaded once."""
        mock_model = MagicMock()
        mock_get_reranker.return_value = mock_model
        mixin = RankingMixin()

        # Access multiple times
        _ = mixin.reranker
        _ = mixin.reranker
        _ = mixin.reranker

        # Should only be loaded once
        assert mock_get_reranker.call_count == 1


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestRankingEdgeCases:
    """Tests for edge cases in ranking."""

    def test_rrf_duplicate_indices(self, ranking_mixin):
        """Test RRF with duplicate indices in same list."""
        dense_results = [(0, 0.9), (0, 0.8)]  # Duplicate
        bm25_results = [(1, 10.0)]

        merged = ranking_mixin.rrf_merge(dense_results, bm25_results)

        # Should handle gracefully
        assert isinstance(merged, list)

    def test_rrf_negative_scores(self, ranking_mixin):
        """Test RRF with negative scores (shouldn't affect RRF)."""
        dense_results = [(0, -0.5), (1, -0.8)]
        bm25_results = [(0, -10.0), (1, -8.0)]

        merged = ranking_mixin.rrf_merge(dense_results, bm25_results)

        # Should still work (RRF only uses rank, not score)
        assert len(merged) > 0

    def test_rrf_very_large_k(self, ranking_mixin):
        """Test RRF with very large k value."""
        dense_results = [(0, 0.9)]
        bm25_results = [(1, 10.0)]

        merged = ranking_mixin.rrf_merge(dense_results, bm25_results, k=1000000)

        assert len(merged) == 2

    @patch('backend.rag.retrieval.ranking.CrossEncoder')
    def test_rerank_single_candidate(self, mock_cross_encoder, ranking_mixin, sample_chunks):
        """Test reranking with single candidate."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]
        mock_cross_encoder.return_value = mock_model
        ranking_mixin._reranker = mock_model

        reranked = ranking_mixin.rerank("query", sample_chunks[:1], top_n=5)

        assert len(reranked) == 1
