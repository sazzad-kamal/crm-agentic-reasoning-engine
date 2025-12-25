"""
Constants for the retrieval module.
"""

from pathlib import Path

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
