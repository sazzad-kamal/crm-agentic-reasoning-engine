"""
Conversation memory utilities.

Provides session clearing and history formatting for LLM prompts.
"""

import logging
from collections import defaultdict
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.agent.nodes.state import Message


logger = logging.getLogger(__name__)

# In-memory storage for session clearing
_memory_store: dict[str, list["Message"]] = defaultdict(list)
_memory_lock = Lock()


def clear_session(session_id: str | None) -> None:
    """
    Clear all messages for a session.

    Args:
        session_id: The session to clear
    """
    if not session_id:
        return

    with _memory_lock:
        if session_id in _memory_store:
            del _memory_store[session_id]
            logger.debug(f"[Memory] Cleared session {session_id}")


def format_history_for_prompt(
    messages: list["Message"],
    max_messages: int = 6,
) -> str:
    """
    Format conversation history for inclusion in LLM prompts.

    Args:
        messages: List of messages
        max_messages: Maximum number of recent messages to include

    Returns:
        Formatted string for prompt inclusion
    """
    if not messages:
        return ""

    recent = messages[-max_messages:]
    lines = []

    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        # Truncate long messages
        if len(content) > 200:
            content = f"{content[:200]}..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


__all__ = [
    "clear_session",
    "format_history_for_prompt",
]
