# backend.rag.retrieval - Retrieval Backends
"""
Retrieval backend implementations for RAG.

Modules:
- base: Base RetrievalBackend with hybrid search (Qdrant + BM25)
- private: PrivateRetrievalBackend for account-scoped retrieval
- embedding: Embedding cache and utilities
- ranking: RRF merging and cross-encoder reranking
"""

from backend.rag.retrieval.base import RetrievalBackend, create_backend
from backend.rag.retrieval.private import PrivateRetrievalBackend, create_private_backend
from backend.rag.retrieval.ranking import RankingMixin

__all__ = [
    "RetrievalBackend",
    "create_backend",
    "PrivateRetrievalBackend",
    "create_private_backend",
    "RankingMixin",
]
