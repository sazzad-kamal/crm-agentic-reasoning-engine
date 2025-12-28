# backend.rag.retrieval - Retrieval Backends
"""
Retrieval backend implementations for RAG.

Modules:
- base: Base RetrievalBackend with hybrid search (Qdrant + BM25)
- private: PrivateRetrievalBackend for account-scoped retrieval
- embedding: Embedding cache and utilities
- ranking: RRF merging and cross-encoder reranking
- langchain_retriever: LangChain-compatible BaseRetriever wrapper
"""

from backend.rag.retrieval.base import RetrievalBackend, create_backend
from backend.rag.retrieval.private import PrivateRetrievalBackend, create_private_backend
from backend.rag.retrieval.ranking import RankingMixin
from backend.rag.retrieval.langchain_retriever import (
    AcmeCRMRetriever,
    create_langchain_retriever,
)
from backend.rag.retrieval.preload import preload_models, is_preloaded

__all__ = [
    "RetrievalBackend",
    "create_backend",
    "PrivateRetrievalBackend",
    "create_private_backend",
    "RankingMixin",
    # LangChain integration
    "AcmeCRMRetriever",
    "create_langchain_retriever",
    # Model preloading
    "preload_models",
    "is_preloaded",
]
