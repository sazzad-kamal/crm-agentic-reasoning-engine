"""
Embedding utilities and caching.

Provides embedding cache and helper functions for vector operations.
"""

import logging

import numpy as np

from backend.rag.config import get_config, EMBEDDING_CACHE_SIZE


logger = logging.getLogger(__name__)


# =============================================================================
# Embedding Cache
# =============================================================================

# Simple in-memory cache for embeddings
_embedding_cache: dict[str, np.ndarray] = {}


def get_cached_embedding(query: str) -> np.ndarray | None:
    """Get cached embedding for a query if it exists."""
    config = get_config()
    if not config.enable_embedding_cache:
        return None
    return _embedding_cache.get(query)


def cache_embedding(query: str, embedding: np.ndarray) -> None:
    """Cache an embedding for a query."""
    config = get_config()
    if not config.enable_embedding_cache:
        return
    
    # Limit cache size
    if len(_embedding_cache) >= EMBEDDING_CACHE_SIZE:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]
    
    _embedding_cache[query] = embedding


def clear_embedding_cache() -> None:
    """Clear the embedding cache."""
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


def get_cache_stats() -> dict:
    """Get embedding cache statistics."""
    return {
        "size": len(_embedding_cache),
        "max_size": get_config().embedding_cache_size,
    }
