"""
Conversation and thread management for agent sessions.

Handles LangGraph checkpointing for conversation persistence.
"""

import logging

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


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
    import uuid

    thread_id = session_id or str(uuid.uuid4())
    return {"configurable": {"thread_id": thread_id}}


__all__ = [
    "get_checkpointer",
    "get_session_state",
    "get_session_messages",
    "build_thread_config",
]
