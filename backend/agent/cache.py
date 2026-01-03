"""
Query caching for agent responses.

Provides in-memory caching with TTL support for repeated identical queries.
"""

import hashlib
import logging
import time

logger = logging.getLogger(__name__)


# Cache configuration
_CACHE_MAX_SIZE = 128
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Cache storage: key -> (result, timestamp)
_query_cache: dict[str, tuple[dict, float]] = {}


def make_cache_key(question: str, mode: str, company_id: str | None) -> str:
    """Generate cache key from query parameters."""
    key_data = f"{question.lower().strip()}|{mode}|{company_id or ''}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def get_cached_result(cache_key: str) -> dict | None:
    """Get cached result if valid (not expired)."""
    if cache_key in _query_cache:
        result, timestamp = _query_cache[cache_key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            logger.debug(f"[Cache] Hit for key {cache_key}")
            return result
        else:
            # Expired, remove from cache
            del _query_cache[cache_key]
            logger.debug(f"[Cache] Expired key {cache_key}")
    return None


def set_cached_result(cache_key: str, result: dict) -> None:
    """Store result in cache with timestamp."""
    # Evict oldest entries if cache is full
    if len(_query_cache) >= _CACHE_MAX_SIZE:
        # Remove oldest entry (first inserted)
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]
        logger.debug(f"[Cache] Evicted oldest key {oldest_key}")

    _query_cache[cache_key] = (result, time.time())
    logger.debug(f"[Cache] Stored key {cache_key}")


def clear_query_cache() -> None:
    """Clear the query cache (useful for testing)."""
    _query_cache.clear()
    logger.debug("[Cache] Cleared all entries")


__all__ = [
    "make_cache_key",
    "get_cached_result",
    "set_cached_result",
    "clear_query_cache",
]
