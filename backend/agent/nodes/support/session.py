"""
Session management for agent conversations.

Provides LangGraph checkpointing and query caching for conversation persistence.
"""

import hashlib
import logging
import time
import uuid

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


# =============================================================================
# Conversation Checkpointing
# =============================================================================

# Global checkpointer for conversation persistence
_checkpointer = MemorySaver()


def get_checkpointer() -> MemorySaver:
    """Get the global checkpointer instance."""
    return _checkpointer


def get_session_state(session_id: str) -> dict | None:
    """
    Get the checkpointed state for a session.

    Args:
        session_id: The session/thread ID

    Returns:
        The stored state dict, or None if not found
    """
    try:
        config = {"configurable": {"thread_id": session_id}}
        checkpoint = _checkpointer.get(config)
        if checkpoint:
            return checkpoint.get("channel_values", {})
    except Exception as e:
        logger.warning(f"Failed to get session state: {e}")
    return None


def get_session_messages(session_id: str) -> list:
    """
    Get conversation messages from a session checkpoint.

    Args:
        session_id: The session/thread ID

    Returns:
        List of Message objects from the checkpoint, or empty list
    """
    state = get_session_state(session_id)
    if state:
        messages = state.get("messages", [])
        if messages:
            logger.debug(f"[Conversation] Loaded {len(messages)} messages from checkpoint")
        return messages
    return []


def build_thread_config(session_id: str | None) -> dict:
    """
    Build LangGraph config with thread_id for checkpointing.

    Args:
        session_id: Optional session ID (generates UUID if None)

    Returns:
        Config dict with thread_id
    """
    thread_id = session_id or str(uuid.uuid4())
    return {"configurable": {"thread_id": thread_id}}


# =============================================================================
# Query Caching
# =============================================================================

# Cache configuration
_CACHE_MAX_SIZE = 128
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Cache storage: key -> (result, timestamp)
_query_cache: dict[str, tuple[dict, float]] = {}


def make_cache_key(question: str) -> str:
    """Generate cache key from question."""
    key_data = question.lower().strip()
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
    # Conversation
    "get_checkpointer",
    "get_session_state",
    "get_session_messages",
    "build_thread_config",
    # Cache
    "make_cache_key",
    "get_cached_result",
    "set_cached_result",
    "clear_query_cache",
]
