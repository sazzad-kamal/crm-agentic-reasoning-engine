"""
RAG configuration constants.

Centralized configuration for Qdrant paths, collection names, and models.
"""

from pathlib import Path

# Directory paths (backend/agent/fetch/rag/config.py -> backend/)
_BACKEND_ROOT = Path(__file__).parent.parent.parent.parent
QDRANT_PATH = _BACKEND_ROOT / "data" / "qdrant"
JSONL_PATH = _BACKEND_ROOT / "data" / "csv" / "texts.jsonl"

# Collection names
TEXT_COLLECTION = "acme_crm_texts"

# Embedding models (ingestion + retrieval)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"

# Retrieval configuration
SEARCH_TOP_K = 30  # Candidates per method (dense + sparse) before reranking

# Reranker configuration
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 5  # Final chunks after reranking


__all__ = [
    "QDRANT_PATH",
    "JSONL_PATH",
    "TEXT_COLLECTION",
    "EMBEDDING_MODEL",
    "SPARSE_EMBEDDING_MODEL",
    "SEARCH_TOP_K",
    "RERANKER_MODEL",
    "RERANKER_TOP_K",
]
