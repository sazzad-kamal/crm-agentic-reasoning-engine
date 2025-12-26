"""
Base retrieval backend with hybrid search (Qdrant + BM25) and cross-encoder reranking.

Components:
- Dense vector index: Qdrant (local, on-disk)
- Sparse index: BM25Okapi from rank_bm25
- Embeddings: sentence-transformers (BAAI/bge-small-en-v1.5)
- Reranker: cross-encoder (ms-marco-MiniLM-L-6-v2)

Usage:
    backend = RetrievalBackend()
    backend.build_indexes(chunks)
    results = backend.retrieve_candidates("my query", k=10)
"""

import logging
from itertools import batched
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Record,
)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from backend.rag.models import DocumentChunk, ScoredChunk
from backend.rag.retrieval.constants import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    EMBEDDING_DIM,
    DOCS_COLLECTION,
    QDRANT_PATH,
)
from backend.rag.retrieval.embedding import get_cached_embedding, cache_embedding
from backend.rag.retrieval.ranking import RankingMixin


logger = logging.getLogger(__name__)


__all__ = [
    "RetrievalBackend",
    "create_backend",
]


# =============================================================================
# Retrieval Backend Class
# =============================================================================

class RetrievalBackend(RankingMixin):
    """
    Hybrid retrieval backend combining dense (Qdrant) and sparse (BM25) search
    with cross-encoder reranking.
    
    Features:
    - Dense vector search via Qdrant
    - Sparse BM25 search
    - RRF (Reciprocal Rank Fusion) for merging results (via RankingMixin)
    - Cross-encoder reranking (via RankingMixin)
    - Embedding caching for repeated queries
    """
    
    def __init__(
        self,
        qdrant_path: Path | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
    ):
        """
        Initialize the retrieval backend.
        
        Args:
            qdrant_path: Path to store Qdrant data (default from config)
            collection_name: Name of the Qdrant collection (default from config)
            embedding_model: Sentence transformer model for embeddings (default from config)
            reranker_model: Cross-encoder model for reranking (default from config)
        """
        self.qdrant_path = qdrant_path or QDRANT_PATH
        self.collection_name = collection_name or DOCS_COLLECTION
        
        # Initialize Qdrant client (local storage)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.qdrant = QdrantClient(path=str(self.qdrant_path))
        logger.info(f"Qdrant client initialized at {self.qdrant_path}")
        
        # Lazy load models
        self._embedding_model_name = embedding_model or EMBEDDING_MODEL
        self._reranker_model_name = reranker_model or RERANKER_MODEL
        self._embedding_model: SentenceTransformer | None = None
        self._reranker = None  # Managed by RankingMixin
        
        # BM25 index (in-memory)
        self._bm25: BM25Okapi | None = None
        self._chunks: list[DocumentChunk] = []
        self._chunk_id_to_idx: dict[str, int] = {}
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self._embedding_model_name}")
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer for BM25."""
        return text.lower().split()
    
    def build_indexes(self, chunks: list[DocumentChunk]) -> None:
        """
        Build both Qdrant and BM25 indexes from chunks.
        
        Args:
            chunks: List of DocumentChunk objects to index
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        self._chunks = chunks
        self._chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}
        
        logger.info(f"Building indexes for {len(chunks)} chunks...")
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        
        # Build Qdrant index
        logger.info("Building Qdrant dense index...")
        
        # Recreate collection
        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)
        
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        
        # Generate embeddings in batches
        batch_size = 32
        texts = [c.text for c in chunks]
        
        all_embeddings = []
        processed = 0
        for batch_texts in batched(texts, batch_size):
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(batch_embeddings)
            processed += len(batch_texts)
            logger.debug(f"Embedded {processed}/{len(texts)} chunks")
        
        # Upload to Qdrant
        points = [
            PointStruct(
                id=i,
                vector=all_embeddings[i].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "text": chunk.text,  # Full text for BM25 retrieval
                    "metadata": chunk.metadata,
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.info(f"Indexes built: {len(chunks)} vectors in Qdrant, BM25 ready")
    
    def _dense_search(
        self,
        query: str,
        k: int = 20,
        qdrant_filter: object | None = None,
        **kwargs,
    ) -> list[tuple[int, float]]:
        """
        Perform dense search using Qdrant.
        
        Args:
            query: Search query
            k: Number of results to return
            qdrant_filter: Optional Qdrant filter object
            **kwargs: Additional arguments (ignored in base class)
        
        Returns:
            List of (chunk_index, score) tuples.
        """
        # Check cache first (only for unfiltered queries)
        cached_embedding = None
        if qdrant_filter is None:
            cached_embedding = get_cached_embedding(query)
        
        if cached_embedding is not None:
            query_embedding = cached_embedding
        else:
            query_embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
            )
            # Cache the embedding (only for unfiltered queries)
            if qdrant_filter is None:
                cache_embedding(query, query_embedding)
        
        # Use query_points (new Qdrant API)
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
            query_filter=qdrant_filter,
            limit=k,
        ).points
        
        logger.debug(f"Dense search returned {len(results)} results")
        return [(hit.id, hit.score) for hit in results]
    
    def _bm25_search(self, query: str, k: int = 20, **kwargs) -> list[tuple[int, float]]:
        """
        Perform BM25 search.
        
        Returns list of (chunk_index, score) tuples.
        """
        if self._bm25 is None:
            raise ValueError("BM25 index not built. Call build_indexes() first.")
        
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def retrieve_candidates(
        self,
        query: str,
        k_dense: int = 20,
        k_bm25: int = 20,
        top_n: int = 10,
        use_reranker: bool = True,
        **kwargs,
    ) -> list[ScoredChunk]:
        """
        Retrieve candidate chunks using hybrid search + reranking.
        
        Args:
            query: The search query
            k_dense: Number of results from dense search
            k_bm25: Number of results from BM25 search
            top_n: Number of final results to return
            use_reranker: Whether to apply cross-encoder reranking
            **kwargs: Additional arguments for subclass search methods
            
        Returns:
            List of ScoredChunk objects with scores
        """
        if not self._chunks:
            raise ValueError("No chunks loaded. Call build_indexes() first.")
        
        # Run both searches (subclasses can override _dense_search/_bm25_search)
        dense_results = self._dense_search(query, k=k_dense, **kwargs)
        bm25_results = self._bm25_search(query, k=k_bm25, **kwargs)
        
        # Build scored results from search outputs
        return self._build_scored_results(
            query=query,
            dense_results=dense_results,
            bm25_results=bm25_results,
            k_dense=k_dense,
            k_bm25=k_bm25,
            top_n=top_n,
            use_reranker=use_reranker,
        )
    
    def _build_scored_results(
        self,
        query: str,
        dense_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
        k_dense: int,
        k_bm25: int,
        top_n: int,
        use_reranker: bool,
    ) -> list[ScoredChunk]:
        """
        Build scored results from search outputs.
        
        Shared logic extracted from retrieve_candidates to avoid duplication.
        """
        # Create score lookup
        dense_scores = dict(dense_results)
        bm25_scores = dict(bm25_results)
        
        # Merge with RRF (uses RankingMixin)
        merged = self.rrf_merge(dense_results, bm25_results)
        
        # Get candidate chunks
        candidates = []
        rrf_scores = {}
        for idx, rrf_score in merged[:k_dense + k_bm25]:
            chunk = self._chunks[idx]
            candidates.append(chunk)
            rrf_scores[chunk.chunk_id] = rrf_score
        
        # Rerank if enabled (uses RankingMixin)
        if use_reranker and candidates:
            reranked = self.rerank(query, candidates, top_n=top_n)
            
            results = []
            for chunk, rerank_score in reranked:
                idx = self._chunk_id_to_idx[chunk.chunk_id]
                scored_chunk = ScoredChunk(
                    chunk=chunk,
                    dense_score=dense_scores.get(idx, 0.0),
                    bm25_score=bm25_scores.get(idx, 0.0),
                    rrf_score=rrf_scores.get(chunk.chunk_id, 0.0),
                    rerank_score=float(rerank_score),
                )
                results.append(scored_chunk)
        else:
            # Return by RRF score only
            results = []
            for idx, rrf_score in merged[:top_n]:
                chunk = self._chunks[idx]
                scored_chunk = ScoredChunk(
                    chunk=chunk,
                    dense_score=dense_scores.get(idx, 0.0),
                    bm25_score=bm25_scores.get(idx, 0.0),
                    rrf_score=rrf_score,
                    rerank_score=0.0,
                )
                results.append(scored_chunk)
        
        return results
    
    def _scroll_all_points(self, batch_size: int = 100) -> list[Record]:
        """
        Scroll through all points in Qdrant collection.

        Args:
            batch_size: Number of points per scroll request

        Returns:
            List of Qdrant Record objects with payloads
        """
        if not self.qdrant.collection_exists(self.collection_name):
            return []
        
        all_points = []
        offset = None
        
        while True:
            results, offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(results)
            
            if offset is None:
                break
        
        return all_points
    
    def load_chunks_from_qdrant(self) -> list[DocumentChunk]:
        """
        Load chunks from Qdrant payloads.
        
        Returns:
            List of DocumentChunk objects reconstructed from Qdrant payloads
        """
        collection_info = self.qdrant.get_collection(self.collection_name) if self.qdrant.collection_exists(self.collection_name) else None
        if collection_info is None or collection_info.points_count == 0:
            return []
        
        # Use shared scroll method
        all_points = self._scroll_all_points()
        
        chunks = []
        for point in all_points:
            payload = point.payload
            chunk = DocumentChunk(
                chunk_id=payload["chunk_id"],
                doc_id=payload["doc_id"],
                title=payload["title"],
                text=payload["text"],
                metadata=payload.get("metadata", {}),
            )
            chunks.append(chunk)
        
        # Sort by point ID to maintain original order
        chunks_by_id = {c.chunk_id: c for c in chunks}
        return list(chunks_by_id.values())


# =============================================================================
# Convenience Functions
# =============================================================================

def create_backend() -> RetrievalBackend:
    """
    Create and initialize a retrieval backend.
    
    Returns:
        Initialized RetrievalBackend
        
    Raises:
        ValueError: If no Qdrant collection exists
    """
    backend = RetrievalBackend()
    
    # Check if collection exists and has vectors
    if backend.qdrant.collection_exists(DOCS_COLLECTION):
        collection_info = backend.qdrant.get_collection(DOCS_COLLECTION)
        if collection_info.points_count > 0:
            logger.info(f"Using existing Qdrant collection with {collection_info.points_count} vectors")
            
            # Load chunks from Qdrant payloads
            chunks = backend.load_chunks_from_qdrant()
            backend._chunks = chunks
            backend._chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}
            
            # Build BM25 only
            logger.info("Building BM25 index from existing chunks...")
            tokenized_corpus = [backend._tokenize(c.text) for c in chunks]
            backend._bm25 = BM25Okapi(tokenized_corpus)
            return backend
    
    raise ValueError(
        "No Qdrant collection found. Run 'python -m backend.rag.ingest.docs' first."
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Retrieval Backend")
    print("=" * 60)
    
    backend = create_backend()
    
    # Test query
    query = "What is an Opportunity in Acme CRM?"
    print(f"\nQuery: {query}")
    print("-" * 40)
    
    results = backend.retrieve_candidates(query, top_n=5)
    
    for i, scored in enumerate(results, 1):
        print(f"\n{i}. [{scored.chunk.doc_id}] (rerank: {scored.rerank_score:.3f})")
        print(f"   {scored.chunk.text[:200]}...")
