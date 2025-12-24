"""
Constants for the retrieval module.
"""

# Embedding & Reranker Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384
EMBEDDING_CACHE_SIZE = 1000

# Qdrant Collection
DOCS_COLLECTION = "acme_crm_docs"

# Retrieval Defaults
DEFAULT_K_DENSE = 20
DEFAULT_K_BM25 = 20
DEFAULT_TOP_N = 10
RRF_K = 60
