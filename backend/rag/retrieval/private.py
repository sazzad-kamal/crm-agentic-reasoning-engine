"""
Private retrieval backend with metadata filtering for account-scoped RAG (MVP2).

Extends the base RetrievalBackend to support:
- Metadata filtering by company_id in Qdrant
- Filtered BM25 search
- Company-scoped hybrid retrieval

Usage:
    backend = PrivateRetrievalBackend()
    backend.load_from_qdrant()
    results = backend.retrieve_candidates("query", company_filter="ACME-MFG")
"""

import logging
import threading
from pathlib import Path
from typing import override

import numpy as np
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)
from rank_bm25 import BM25Okapi

from backend.rag.models import DocumentChunk, ScoredChunk
from backend.rag.retrieval.constants import PRIVATE_COLLECTION
from backend.rag.retrieval.base import RetrievalBackend


logger = logging.getLogger(__name__)


__all__ = [
    "PrivateRetrievalBackend",
    "create_private_backend",
    "clear_private_backend_cache",
]


class PrivateRetrievalBackend(RetrievalBackend):
    """
    Hybrid retrieval backend for private CRM texts with metadata filtering.
    
    Inherits from RetrievalBackend and adds:
    - Company-specific filtering in Qdrant dense search
    - Company-specific BM25 indexes (built on demand)
    - Company-scoped hybrid retrieval
    """
    
    def __init__(
        self,
        qdrant_path: Path | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
    ):
        """Initialize the private retrieval backend."""
        # Use private collection by default
        collection_name = collection_name or PRIVATE_COLLECTION
        
        # Initialize base class
        super().__init__(
            qdrant_path=qdrant_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
        
        # Company-specific BM25 indexes (built on demand)
        self._company_bm25: dict[str, tuple[BM25Okapi | None, list[int]]] = {}
    
    @override
    def load_from_qdrant(self) -> None:
        """
        Load chunks from existing Qdrant collection.
        
        Rebuilds BM25 index from stored payloads.
        """
        if not self.qdrant.collection_exists(self.collection_name):
            raise ValueError(
                f"Collection '{self.collection_name}' not found. "
                "Run 'python -m backend.rag.ingest.private_text' first."
            )
        
        collection_info = self.qdrant.get_collection(self.collection_name)
        num_points = collection_info.points_count
        
        if num_points == 0:
            raise ValueError(f"Collection '{self.collection_name}' is empty.")
        
        logger.info(f"Loading {num_points} points from '{self.collection_name}'...")
        
        # Use shared scroll method
        all_points = self._scroll_all_points()
        
        # Convert to DocumentChunks with private metadata
        self._chunks = []
        for point in all_points:
            payload = point.payload
            chunk = DocumentChunk(
                chunk_id=payload.get("chunk_id", str(point.id)),
                doc_id=payload.get("doc_id", ""),
                title=payload.get("title", ""),
                text=payload.get("text", ""),
                metadata={
                    "company_id": payload.get("company_id", ""),
                    "type": payload.get("type", ""),
                    "source_id": payload.get("source_id", ""),
                    "contact_id": payload.get("contact_id"),
                    "opportunity_id": payload.get("opportunity_id"),
                    "qdrant_id": point.id,  # Store Qdrant point ID
                },
            )
            self._chunks.append(chunk)
        
        self._chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        
        # Build global BM25 index
        logger.info(f"Building BM25 index for {len(self._chunks)} chunks...")
        tokenized = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        
        logger.info(f"Loaded {len(self._chunks)} chunks from Qdrant")
    
    def _get_company_bm25(self, company_id: str) -> tuple[BM25Okapi | None, list[int]]:
        """
        Get or build company-specific BM25 index.
        
        Returns:
            (BM25Okapi index, list of original chunk indices)
        """
        if company_id in self._company_bm25:
            return self._company_bm25[company_id]
        
        # Find chunks for this company
        indices = []
        texts = []
        for i, chunk in enumerate(self._chunks):
            if chunk.metadata.get("company_id") == company_id:
                indices.append(i)
                texts.append(self._tokenize(chunk.text))
        
        if not texts:
            # Return empty index
            self._company_bm25[company_id] = (None, [])
            return None, []
        
        bm25 = BM25Okapi(texts)
        self._company_bm25[company_id] = (bm25, indices)
        return bm25, indices
    
    def _build_company_filter(self, company_id: str) -> Filter:
        """Build a Qdrant filter for a specific company."""
        return Filter(
            must=[
                FieldCondition(
                    key="company_id",
                    match=MatchValue(value=company_id),
                )
            ]
        )
    
    @override
    def _dense_search(
        self,
        query: str,
        k: int = 20,
        company_filter: str | None = None,
        qdrant_filter: object | None = None,
        **kwargs,
    ) -> list[tuple[int, float]]:
        """
        Perform dense search using Qdrant with optional company filter.
        
        Extends base class to support company_filter parameter.
        
        Returns list of (chunk_index, score) tuples.
        """
        # Build filter if company specified
        if company_filter:
            qdrant_filter = self._build_company_filter(company_filter)
            # For company-filtered queries, query directly and map via chunk_id
            query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            qdrant_results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                query_filter=qdrant_filter,
                limit=k,
            ).points
            output = []
            for hit in qdrant_results:
                chunk_id = hit.payload.get("chunk_id")
                if chunk_id in self._chunk_id_to_idx:
                    idx = self._chunk_id_to_idx[chunk_id]
                    output.append((idx, hit.score))
            return output
        
        # No filter - delegate to base class
        return super()._dense_search(query, k=k, qdrant_filter=qdrant_filter)
    
    @override
    def _bm25_search(
        self,
        query: str,
        k: int = 20,
        company_filter: str | None = None,
        **kwargs,
    ) -> list[tuple[int, float]]:
        """
        Perform BM25 search with optional company filter.
        
        Extends base class to support company-specific BM25 indexes.
        
        Returns list of (chunk_index, score) tuples.
        """
        if company_filter:
            # Use company-specific BM25 index
            bm25, indices = self._get_company_bm25(company_filter)
            if bm25 is None:
                return []
            
            tokenized_query = self._tokenize(query)
            scores = bm25.get_scores(tokenized_query)
            
            # Get top k, map back to original indices
            top_local = np.argsort(scores)[::-1][:k]
            return [(indices[i], float(scores[i])) for i in top_local if scores[i] > 0]
        
        # No filter - delegate to base class
        return super()._bm25_search(query, k=k)
    
    @override
    def retrieve_candidates(
        self,
        query: str,
        k_dense: int = 20,
        k_bm25: int = 20,
        top_n: int = 10,
        use_reranker: bool = True,
        company_filter: str | None = None,
    ) -> list[ScoredChunk]:
        """
        Retrieve candidate chunks with optional company filtering.
        
        Args:
            query: Search query
            k_dense: Number of dense search results
            k_bm25: Number of BM25 search results
            top_n: Final number of results
            use_reranker: Whether to apply cross-encoder reranking
            company_filter: If set, only retrieve chunks for this company_id
            
        Returns:
            List of ScoredChunk objects
        """
        if not self._chunks:
            raise ValueError("No chunks loaded. Call load_from_qdrant() first.")
        
        # Delegate to base class with company_filter passed through
        return super().retrieve_candidates(
            query=query,
            k_dense=k_dense,
            k_bm25=k_bm25,
            top_n=top_n,
            use_reranker=use_reranker,
            company_filter=company_filter,
        )
    
    def get_all_companies(self) -> list[str]:
        """Get list of all company_ids in the index."""
        companies = set()
        for chunk in self._chunks:
            cid = chunk.metadata.get("company_id")
            if cid:
                companies.add(cid)
        return sorted(companies)


# =============================================================================
# Factory Function
# =============================================================================

# Module-level singleton cache for thread-safe backend access
_private_backend_cache: PrivateRetrievalBackend | None = None
_private_backend_lock = threading.Lock()


def create_private_backend(
    rebuild: bool = False,
    use_cache: bool = True,
) -> PrivateRetrievalBackend:
    """
    Create and initialize a private retrieval backend.

    Uses a singleton pattern with thread-safe initialization to prevent
    Qdrant concurrent access errors when running in parallel (e.g., evals).

    Args:
        rebuild: If True, force rebuild from CSV (not implemented, use ingest script)
        use_cache: If True (default), return cached singleton instance.
                   If False, create a new instance (use for testing).

    Returns:
        Initialized PrivateRetrievalBackend
    """
    global _private_backend_cache

    # Return cached instance if available and caching enabled
    if use_cache and _private_backend_cache is not None:
        return _private_backend_cache

    with _private_backend_lock:
        # Double-check pattern: re-check after acquiring lock
        if use_cache and _private_backend_cache is not None:
            return _private_backend_cache

        backend = PrivateRetrievalBackend()

        if not backend.qdrant.collection_exists(PRIVATE_COLLECTION):
            raise ValueError(
                f"Private collection '{PRIVATE_COLLECTION}' not found. "
                "Run 'python -m backend.rag.ingest.private_text' first."
            )

        backend.load_from_qdrant()

        # Cache the singleton if caching enabled
        if use_cache:
            _private_backend_cache = backend

        return backend


def clear_private_backend_cache() -> None:
    """Clear the cached private backend singleton (useful for testing)."""
    global _private_backend_cache
    with _private_backend_lock:
        _private_backend_cache = None
        logger.debug("Private backend cache cleared")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Private Retrieval Backend")
    print("=" * 60)
    
    backend = create_private_backend()
    
    print(f"\nCompanies in index: {backend.get_all_companies()}")
    
    # Test unfiltered query
    query = "What happened with the renewal?"
    print(f"\nQuery (unfiltered): {query}")
    results = backend.retrieve_candidates(query, top_n=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.chunk.metadata.get('company_id')}] {r.chunk.doc_id[:40]}...")
    
    # Test filtered query
    company = "ACME-MFG"
    print(f"\nQuery (filtered to {company}): {query}")
    results = backend.retrieve_candidates(query, top_n=3, company_filter=company)
    for i, r in enumerate(results, 1):
        cid = r.chunk.metadata.get('company_id')
        print(f"  {i}. [{cid}] {r.chunk.doc_id[:40]}... (company match: {cid == company})")
