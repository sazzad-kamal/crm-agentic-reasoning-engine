"""
Retrieval backend with hybrid search (Qdrant + BM25) and cross-encoder reranking.

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

import json
from pathlib import Path
from typing import Optional
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from project1_rag.doc_models import DocumentChunk, ScoredChunk
from project1_rag.ingest_docs import load_chunks, OUTPUT_FILE


# =============================================================================
# Configuration
# =============================================================================

QDRANT_PATH = Path("data/qdrant")
COLLECTION_NAME = "acme_crm_docs"

# Model names
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Embedding dimension for bge-small-en-v1.5
EMBEDDING_DIM = 384


# =============================================================================
# Retrieval Backend Class
# =============================================================================

class RetrievalBackend:
    """
    Hybrid retrieval backend combining dense (Qdrant) and sparse (BM25) search
    with cross-encoder reranking.
    """
    
    def __init__(
        self,
        qdrant_path: Path = QDRANT_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
    ):
        """
        Initialize the retrieval backend.
        
        Args:
            qdrant_path: Path to store Qdrant data
            collection_name: Name of the Qdrant collection
            embedding_model: Sentence transformer model for embeddings
            reranker_model: Cross-encoder model for reranking
        """
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        
        # Initialize Qdrant client (local storage)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.qdrant = QdrantClient(path=str(self.qdrant_path))
        
        # Lazy load models
        self._embedding_model_name = embedding_model
        self._reranker_model_name = reranker_model
        self._embedding_model: Optional[SentenceTransformer] = None
        self._reranker: Optional[CrossEncoder] = None
        
        # BM25 index (in-memory)
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: list[DocumentChunk] = []
        self._chunk_id_to_idx: dict[str, int] = {}
    
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
        
        print(f"Building indexes for {len(chunks)} chunks...")
        
        # Build BM25 index
        print("  Building BM25 index...")
        tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        
        # Build Qdrant index
        print("  Building Qdrant dense index...")
        
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
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(batch_embeddings)
            print(f"    Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        # Upload to Qdrant
        points = [
            PointStruct(
                id=i,
                vector=all_embeddings[i].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "text": chunk.text[:500],  # Store truncated text for debugging
                    "metadata": chunk.metadata,
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        print(f"  Indexes built: {len(chunks)} vectors in Qdrant, BM25 ready")
    
    def _dense_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """
        Perform dense search using Qdrant.
        
        Returns list of (chunk_index, score) tuples.
        """
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        )
        
        # Use query_points for qdrant-client >= 1.16
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=k,
        ).points
        
        return [(hit.id, hit.score) for hit in results]
    
    def _bm25_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
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
    
    def _rrf_merge(
        self,
        dense_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        
        Args:
            dense_results: Results from dense search
            bm25_results: Results from BM25 search
            k: RRF constant (default 60)
            
        Returns:
            Merged list of (chunk_index, rrf_score) tuples
        """
        scores = {}
        
        # Add dense scores
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        
        # Add BM25 scores
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        
        # Sort by RRF score
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return merged
    
    def _rerank(
        self,
        query: str,
        candidates: list[DocumentChunk],
        top_n: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Rerank candidates using cross-encoder.
        
        Returns list of (chunk, rerank_score) tuples.
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
    
    def retrieve_candidates(
        self,
        query: str,
        k_dense: int = 20,
        k_bm25: int = 20,
        top_n: int = 10,
        use_reranker: bool = True,
    ) -> list[ScoredChunk]:
        """
        Retrieve candidate chunks using hybrid search + reranking.
        
        Args:
            query: The search query
            k_dense: Number of results from dense search
            k_bm25: Number of results from BM25 search
            top_n: Number of final results to return
            use_reranker: Whether to apply cross-encoder reranking
            
        Returns:
            List of ScoredChunk objects with scores
        """
        if not self._chunks:
            raise ValueError("No chunks loaded. Call build_indexes() first.")
        
        # Run both searches
        dense_results = self._dense_search(query, k=k_dense)
        bm25_results = self._bm25_search(query, k=k_bm25)
        
        # Create score lookup
        dense_scores = {idx: score for idx, score in dense_results}
        bm25_scores = {idx: score for idx, score in bm25_results}
        
        # Merge with RRF
        merged = self._rrf_merge(dense_results, bm25_results)
        
        # Get candidate chunks
        candidates = []
        rrf_scores = {}
        for idx, rrf_score in merged[:k_dense + k_bm25]:
            chunk = self._chunks[idx]
            candidates.append(chunk)
            rrf_scores[chunk.chunk_id] = rrf_score
        
        # Rerank if enabled
        if use_reranker and candidates:
            reranked = self._rerank(query, candidates, top_n=top_n)
            
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
    
    def load_and_build(self, chunks_path: Path = OUTPUT_FILE) -> None:
        """
        Load chunks from disk and build indexes.
        
        Args:
            chunks_path: Path to the chunks Parquet file
        """
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}. "
                "Run 'python -m project1_rag.ingest_docs' first."
            )
        
        print(f"Loading chunks from {chunks_path}...")
        chunks = load_chunks(chunks_path)
        self.build_indexes(chunks)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a chunk by its ID."""
        idx = self._chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self._chunks[idx]
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def create_backend(rebuild: bool = False) -> RetrievalBackend:
    """
    Create and initialize a retrieval backend.
    
    Args:
        rebuild: If True, always rebuild indexes even if they exist
        
    Returns:
        Initialized RetrievalBackend
    """
    backend = RetrievalBackend()
    
    # Check if collection exists and has vectors
    if not rebuild and backend.qdrant.collection_exists(COLLECTION_NAME):
        collection_info = backend.qdrant.get_collection(COLLECTION_NAME)
        if collection_info.points_count > 0:
            print(f"Using existing Qdrant collection with {collection_info.points_count} vectors")
            # Still need to load chunks for BM25
            chunks = load_chunks()
            backend._chunks = chunks
            backend._chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}
            
            # Build BM25 only
            tokenized_corpus = [backend._tokenize(c.text) for c in chunks]
            backend._bm25 = BM25Okapi(tokenized_corpus)
            return backend
    
    # Build from scratch
    backend.load_and_build()
    return backend


if __name__ == "__main__":
    # Test the retrieval backend
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
