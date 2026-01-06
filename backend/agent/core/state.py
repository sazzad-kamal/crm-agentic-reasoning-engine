"""
Agent state definition for LangGraph workflow.

This module defines the central state structure that flows through the graph.
"""

import logging
from typing import Any, TypedDict, Annotated
from operator import add

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class Source(BaseModel):
    """A source reference for citations."""

    type: str  # "company", "doc", "activity", "opportunity", "history"
    id: str
    label: str


class Message(TypedDict):
    """A single message in conversation history."""

    role: str  # "user" or "assistant"
    content: str


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


class AgentState(TypedDict, total=False):
    """
    State that flows through the LangGraph workflow.

    This is the central data structure that each node can read/write.
    """

    # Input
    question: str

    # Conversation history (persisted by LangGraph checkpointer)
    messages: list[Message]
    conversation_history: str  # Formatted once in route_node, reused by other nodes

    # Router output (flattened from RouterResult)
    mode_used: str
    resolved_company_id: str | None
    company_name_query: str | None  # For resolving company names
    days: int
    intent: str
    owner: str | None  # Role-based owner for filtering

    # Data outputs
    company_data: dict[str, Any] | None
    activities_data: dict[str, Any] | None
    history_data: dict[str, Any] | None
    pipeline_data: dict[str, Any] | None
    renewals_data: dict[str, Any] | None
    contacts_data: dict[str, Any] | None
    groups_data: dict[str, Any] | None
    attachments_data: dict[str, Any] | None

    # Docs output
    docs_answer: str
    docs_sources: list[Source]

    # Account RAG output (private CRM text: notes, attachments)
    account_context_answer: str
    account_context_sources: list[Source]

    # Sources accumulated from all steps (using reducer to append)
    sources: Annotated[list[Source], add]

    # Final outputs
    answer: str
    follow_up_suggestions: list[str]

    # Raw data for UI
    raw_data: dict[str, Any]

    # Error handling
    error: str | None

    # Progress/latency tracking
    steps: list[dict[str, Any]]
    router_latency_ms: int
    answer_latency_ms: int
    llm_latency_ms: int
    followup_latency_ms: int


__all__ = ["Source", "AgentState", "Message", "format_history_for_prompt"]
