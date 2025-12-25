"""
Ranking utilities for retrieval results.

Contains:
- RRF (Reciprocal Rank Fusion) merging
- Cross-encoder reranking
"""

import logging

from sentence_transformers import CrossEncoder

from backend.common.models import DocumentChunk
from backend.rag.retrieval.constants import RRF_K, RERANKER_MODEL


logger = logging.getLogger(__name__)


class RankingMixin:
    """
    Mixin class providing ranking functionality for retrieval backends.
    
    Includes:
    - RRF (Reciprocal Rank Fusion) for merging dense and sparse results
    - Cross-encoder reranking for final result refinement
    """
    
    _reranker_model_name: str = RERANKER_MODEL
    _reranker: CrossEncoder | None = None
    
    @property
    def reranker(self) -> CrossEncoder:
        """Lazy load the reranker model."""
        if self._reranker is None:
            logger.info(f"Loading reranker model: {self._reranker_model_name}")
            self._reranker = CrossEncoder(self._reranker_model_name)
        return self._reranker
    
    def rrf_merge(
        self,
        dense_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
        k: int = RRF_K,
    ) -> list[tuple[int, float]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        
        Args:
            dense_results: Results from dense search as (index, score) tuples
            bm25_results: Results from BM25 search as (index, score) tuples
            k: RRF constant (default 60)
            
        Returns:
            Merged list of (chunk_index, rrf_score) tuples sorted by score
        """
        scores: dict[int, float] = {}
        
        # Add dense scores
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        
        # Add BM25 scores
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        
        # Sort by RRF score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def rerank(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_n: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: The search query
            candidates: List of candidate chunks to rerank
            top_n: Number of top results to return
        
        Returns:
            List of (chunk, rerank_score) tuples sorted by score
        """
        if not candidates:
            return []
        
        # Create query-document pairs
        pairs = [(query, c.text) for c in candidates]
        
        # Score with cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_n]
