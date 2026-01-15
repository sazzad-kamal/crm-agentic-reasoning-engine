"""
Agent state definition for LangGraph workflow.

This module defines the central state structure that flows through the graph.
"""

import logging
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class Message(TypedDict):
    """A single message in conversation history."""

    role: str  # "user" or "assistant"
    content: str


def format_conversation_for_prompt(
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


class AgentState(TypedDict, total=False):
    """
    State that flows through the LangGraph workflow.

    This is the central data structure that each node can read/write.
    Only includes fields needed by workflow nodes or final output.
    Eval-specific data (sql_plan, account_chunks, etc.) is captured
    out-of-band via backend.eval.callback.
    """

    # Input
    question: str

    # Conversation history (persisted by LangGraph checkpointer)
    messages: list[Message]

    # SQL results from fetch node
    sql_results: dict[str, Any]

    # Account RAG output (private CRM text: notes, descriptions)
    account_context_answer: str

    # Final outputs
    answer: str
    follow_up_suggestions: list[str]

    # Error handling
    error: str | None


__all__ = ["AgentState", "Message", "format_conversation_for_prompt"]
