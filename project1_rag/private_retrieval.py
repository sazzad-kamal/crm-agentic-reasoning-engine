"""
Private retrieval backend with metadata filtering for account-scoped RAG (MVP2).

Extends the base retrieval to support:
- Metadata filtering by company_id in Qdrant
- Filtered BM25 search
- Company-scoped hybrid retrieval

Usage:
    backend = PrivateRetrievalBackend()
    backend.load_from_qdrant()
    results = backend.retrieve_candidates("query", company_filter="ACME-MFG")
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from project1_rag.doc_models import DocumentChunk, ScoredChunk
from project1_rag.ingest_private_text import (
    PRIVATE_COLLECTION_NAME,
    QDRANT_PATH,
    EMBEDDING_MODEL,
)
from project1_rag.retrieval_backend import RERANKER_MODEL


# =============================================================================
# Private Retrieval Backend
# =============================================================================

class PrivateRetrievalBackend:
    """
    Hybrid retrieval backend for private CRM texts with metadata filtering.
    
    Supports filtering by company_id for account-scoped retrieval.
    """
    
    def __init__(
        self,
        qdrant_path: Path = QDRANT_PATH,
        collection_name: str = PRIVATE_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
    ):
        """Initialize the private retrieval backend."""
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.qdrant = QdrantClient(path=str(self.qdrant_path))
        
        # Lazy load models
        self._embedding_model_name = embedding_model
        self._reranker_model_name = reranker_model
        self._embedding_model: Optional[SentenceTransformer] = None
        self._reranker: Optional[CrossEncoder] = None
        
        # In-memory chunk storage and BM25 index
        self._chunks: list[DocumentChunk] = []
        self._chunk_id_to_idx: dict[str, int] = {}
        self._bm25: Optional[BM25Okapi] = None
        
        # Company-specific BM25 indexes (built on demand)
        self._company_bm25: dict[str, tuple[BM25Okapi, list[int]]] = {}
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            print(f"Loading embedding model: {self._embedding_model_name}")
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model
    
    @property
    def reranker(self) -> CrossEncoder:
        """Lazy load the reranker model."""
        if self._reranker is None:
            print(f"Loading reranker model: {self._reranker_model_name}")
            self._reranker = CrossEncoder(self._reranker_model_name)
        return self._reranker
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer for BM25."""
        return text.lower().split()
    
    def load_from_qdrant(self) -> None:
        """
        Load chunks from existing Qdrant collection.
        
        Rebuilds BM25 index from stored payloads.
        """
        if not self.qdrant.collection_exists(self.collection_name):
            raise ValueError(
                f"Collection '{self.collection_name}' not found. "
                "Run 'python -m project1_rag.ingest_private_text' first."
            )
        
        collection_info = self.qdrant.get_collection(self.collection_name)
        num_points = collection_info.points_count
        
        if num_points == 0:
            raise ValueError(f"Collection '{self.collection_name}' is empty.")
        
        print(f"Loading {num_points} points from '{self.collection_name}'...")
        
        # Scroll through all points
        all_points = []
        offset = None
        batch_size = 100
        
        while True:
            result = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            all_points.extend(points)
            
            if next_offset is None or len(points) == 0:
                break
            offset = next_offset
        
        # Convert to DocumentChunks
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
        print(f"Building BM25 index for {len(self._chunks)} chunks...")
        tokenized = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        
        print(f"Loaded {len(self._chunks)} chunks from Qdrant")
    
    def _get_company_bm25(self, company_id: str) -> tuple[BM25Okapi, list[int]]:
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
    
    def _dense_search(
        self,
        query: str,
        k: int = 20,
        company_filter: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """
        Perform dense search using Qdrant with optional company filter.
        
        Returns list of (chunk_index, score) tuples.
        """
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        )
        
        # Build filter if company specified
        qdrant_filter = None
        if company_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="company_id",
                        match=MatchValue(value=company_filter),
                    )
                ]
            )
        
        # Use query_points for qdrant-client >= 1.16
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=qdrant_filter,
            limit=k,
        ).points
        
        # Map Qdrant IDs back to chunk indices
        output = []
        for hit in results:
            # Find chunk by qdrant_id or chunk_id from payload
            chunk_id = hit.payload.get("chunk_id")
            if chunk_id in self._chunk_id_to_idx:
                idx = self._chunk_id_to_idx[chunk_id]
                output.append((idx, hit.score))
        
        return output
    
    def _bm25_search(
        self,
        query: str,
        k: int = 20,
        company_filter: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """
        Perform BM25 search with optional company filter.
        
        Returns list of (chunk_index, score) tuples.
        """
        if company_filter:
            # Use company-specific BM25
            bm25, indices = self._get_company_bm25(company_filter)
            if bm25 is None:
                return []
            
            tokenized_query = self._tokenize(query)
            scores = bm25.get_scores(tokenized_query)
            
            # Get top k, map back to original indices
            top_local = np.argsort(scores)[::-1][:k]
            return [(indices[i], float(scores[i])) for i in top_local if scores[i] > 0]
        
        else:
            # Use global BM25
            if self._bm25 is None:
                return []
            
            tokenized_query = self._tokenize(query)
            scores = self._bm25.get_scores(tokenized_query)
            
            top_indices = np.argsort(scores)[::-1][:k]
            return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
    
    def _rrf_merge(
        self,
        dense_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """Merge results using Reciprocal Rank Fusion."""
        scores = {}
        
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return merged
    
    def _rerank(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_n: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []
        
        pairs = [(query, c.text) for c in candidates]
        scores = self.reranker.predict(pairs)
        
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_n]
    
    def retrieve_candidates(
        self,
        query: str,
        k_dense: int = 20,
        k_bm25: int = 20,
        top_n: int = 10,
        use_reranker: bool = True,
        company_filter: Optional[str] = None,
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
        
        # Run searches with filter
        dense_results = self._dense_search(query, k=k_dense, company_filter=company_filter)
        bm25_results = self._bm25_search(query, k=k_bm25, company_filter=company_filter)
        
        # Score lookups
        dense_scores = {idx: score for idx, score in dense_results}
        bm25_scores = {idx: score for idx, score in bm25_results}
        
        # Merge with RRF
        merged = self._rrf_merge(dense_results, bm25_results)
        
        # Get candidates
        candidates = []
        rrf_scores = {}
        for idx, rrf_score in merged[:k_dense + k_bm25]:
            chunk = self._chunks[idx]
            candidates.append(chunk)
            rrf_scores[chunk.chunk_id] = rrf_score
        
        # Rerank
        if use_reranker and candidates:
            reranked = self._rerank(query, candidates, top_n=top_n)
            
            results = []
            for chunk, rerank_score in reranked:
                idx = self._chunk_id_to_idx[chunk.chunk_id]
                scored = ScoredChunk(
                    chunk=chunk,
                    dense_score=dense_scores.get(idx, 0.0),
                    bm25_score=bm25_scores.get(idx, 0.0),
                    rrf_score=rrf_scores.get(chunk.chunk_id, 0.0),
                    rerank_score=float(rerank_score),
                )
                results.append(scored)
        else:
            results = []
            for idx, rrf_score in merged[:top_n]:
                chunk = self._chunks[idx]
                scored = ScoredChunk(
                    chunk=chunk,
                    dense_score=dense_scores.get(idx, 0.0),
                    bm25_score=bm25_scores.get(idx, 0.0),
                    rrf_score=rrf_score,
                    rerank_score=0.0,
                )
                results.append(scored)
        
        return results
    
    def get_chunks_for_company(self, company_id: str) -> list[DocumentChunk]:
        """Get all chunks for a specific company."""
        return [c for c in self._chunks if c.metadata.get("company_id") == company_id]
    
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

def create_private_backend(rebuild: bool = False) -> PrivateRetrievalBackend:
    """
    Create and initialize a private retrieval backend.
    
    Args:
        rebuild: If True, force rebuild from CSV (not implemented, use ingest script)
        
    Returns:
        Initialized PrivateRetrievalBackend
    """
    backend = PrivateRetrievalBackend()
    
    if not backend.qdrant.collection_exists(PRIVATE_COLLECTION_NAME):
        raise ValueError(
            f"Private collection '{PRIVATE_COLLECTION_NAME}' not found. "
            "Run 'python -m project1_rag.ingest_private_text' first."
        )
    
    backend.load_from_qdrant()
    return backend


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
