# backend.rag - RAG Pipeline for Acme CRM
"""
Retrieval-Augmented Generation (RAG) pipeline for CRM documentation.

Modules:
- config: Centralized configuration
- models: Document and chunk models
- utils: Utilities (tokenization, chunking)
- pipeline: Main RAG pipeline
- retrieval: Hybrid search backend (Qdrant + BM25)
- private: Private text retrieval for accounts
- account: Account-scoped RAG
- audit: Audit logging

Subpackages:
- ingest: Document ingestion scripts
"""

from backend.rag.config import get_config, RAGConfig
from backend.rag.utils import estimate_tokens, preprocess_query, recursive_split

__all__ = [
    "get_config",
    "RAGConfig",
    "estimate_tokens",
    "preprocess_query",
    "recursive_split",
]
