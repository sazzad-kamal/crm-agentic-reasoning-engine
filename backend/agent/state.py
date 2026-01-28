"""Agent state definition for LangGraph workflow."""

import logging
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)


def format_conversation_for_prompt(messages: list) -> str:
    """Format conversation history for LLM prompts."""
    if not messages:
        return ""

    lines = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")

    return "\n".join(lines)


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph workflow."""

    # Input
    question: str

    # Conversation history (reducer appends new messages)
    messages: Annotated[list, add_messages]

    # SQL results from fetch node (includes notes columns)
    sql_results: dict[str, Any]

    # Final outputs
    answer: str
    follow_up_suggestions: list[str]
    suggested_action: str | None  # Actionable next step

    # Error handling
    error: str | None


__all__ = ["AgentState", "format_conversation_for_prompt"]
