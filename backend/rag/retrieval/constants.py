"""
Constants for the retrieval module.
"""

import threading
from pathlib import Path

from qdrant_client import QdrantClient

# Paths
_BACKEND_ROOT = Path(__file__).parent.parent.parent
QDRANT_PATH = _BACKEND_ROOT / "data/qdrant"

# Embedding & Reranker Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384
EMBEDDING_CACHE_SIZE = 1000

# Qdrant Collections
DOCS_COLLECTION = "acme_crm_docs"
PRIVATE_COLLECTION = "acme_crm_private"

# Retrieval Defaults
RRF_K = 60


# =============================================================================
# Shared Qdrant Client (Singleton)
# =============================================================================

# Qdrant local mode locks the entire folder - only ONE client can access it.
# This singleton ensures all backends share the same client instance.
_qdrant_client: QdrantClient | None = None
_qdrant_client_lock = threading.Lock()


def get_shared_qdrant_client() -> QdrantClient:
    """
    Get the shared QdrantClient instance (singleton).

    Qdrant local mode locks the storage folder, so we MUST share
    a single client instance across all backends (docs, private, etc.)
    """
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    with _qdrant_client_lock:
        if _qdrant_client is not None:
            return _qdrant_client

        # Ensure path exists
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(QDRANT_PATH))
        return _qdrant_client


def close_shared_qdrant_client() -> None:
    """Close the shared QdrantClient (useful for testing/cleanup)."""
    global _qdrant_client
    with _qdrant_client_lock:
        if _qdrant_client is not None:
            _qdrant_client.close()
            _qdrant_client = None
