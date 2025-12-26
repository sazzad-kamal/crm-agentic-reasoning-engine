"""
Embedding utilities and caching.

Provides embedding cache and helper functions for vector operations.
"""

import logging

import numpy as np
from cachetools import LRUCache

from backend.rag.retrieval.constants import EMBEDDING_CACHE_SIZE


logger = logging.getLogger(__name__)


__all__ = [
    "get_cached_embedding",
    "cache_embedding",
]


# =============================================================================
# Embedding Cache
# =============================================================================

# Thread-safe LRU cache for embeddings with automatic eviction
_embedding_cache: LRUCache[str, np.ndarray] = LRUCache(maxsize=EMBEDDING_CACHE_SIZE)


def get_cached_embedding(query: str) -> np.ndarray | None:
    """
    Get cached embedding for a query if it exists.

    Args:
        query: The query text to look up

    Returns:
        Cached embedding array if found, None otherwise
    """
    return _embedding_cache.get(query)


def cache_embedding(query: str, embedding: np.ndarray) -> None:
    """
    Cache an embedding for a query.

    Uses LRU eviction policy - least recently used embeddings
    are automatically removed when cache reaches max size.

    Args:
        query: The query text to cache
        embedding: The embedding vector to store
    """
    _embedding_cache[query] = embedding
