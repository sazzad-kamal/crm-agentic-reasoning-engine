"""
RAG configuration constants.

Centralized configuration for Qdrant paths, collection names, and models.
"""

from pathlib import Path

# Directory paths
_BACKEND_ROOT = Path(__file__).parent.parent.parent
QDRANT_PATH = _BACKEND_ROOT / "data" / "qdrant"
JSONL_PATH = _BACKEND_ROOT / "data" / "csv" / "private_texts.jsonl"

# Collection names
PRIVATE_COLLECTION = "acme_crm_private"

# Model configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Reranker configuration
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, good quality
RERANKER_TOP_K = 5       # Final chunks after reranking
RETRIEVAL_TOP_K = 15     # Over-retrieve before reranking
RERANKER_ENABLED = True  # Feature flag for easy rollback


__all__ = [
    "QDRANT_PATH",
    "JSONL_PATH",
    "PRIVATE_COLLECTION",
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "RERANKER_TOP_K",
    "RETRIEVAL_TOP_K",
    "RERANKER_ENABLED",
]
