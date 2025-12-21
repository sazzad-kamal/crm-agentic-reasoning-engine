# project1_rag - Docs-only RAG experiment for Acme CRM
"""
A self-contained RAG experiment that:
- Ingests and chunks Markdown docs
- Builds hybrid retrieval (Qdrant + BM25)
- Implements query rewrite + HyDE
- Provides evaluation harness with RAG triad metrics

Key modules:
- config: Centralized configuration
- utils: Shared utilities (token estimation, chunking)
- rag_pipeline: Main RAG pipeline
- retrieval_backend: Hybrid search backend
"""

from project1_rag.config import get_config, RAGConfig
from project1_rag.utils import estimate_tokens, preprocess_query, recursive_split

__all__ = [
    "get_config",
    "RAGConfig",
    "estimate_tokens",
    "preprocess_query",
    "recursive_split",
]
