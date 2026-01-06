"""
Agent state definition for LangGraph workflow.

This module defines the central state structure that flows through the graph.
"""

import logging
from typing import Any, TypedDict, Annotated
from operator import add

from backend.agent.core.schemas import Source, RouterResult


logger = logging.getLogger(__name__)


class Message(TypedDict):
    """A single message in conversation history."""

    role: str  # "user" or "assistant"
    content: str
    company_id: str | None  # Company context for this message


class AgentState(TypedDict, total=False):
    """
    State that flows through the LangGraph workflow.

    This is the central data structure that each node can read/write.
    """

    # Input
    question: str
    mode: str  # "auto", "docs", "data", "data+docs"
    company_id: str | None
    session_id: str | None
    user_id: str | None

    # Conversation history (loaded from memory at start)
    messages: list[Message]

    # Router output
    router_result: RouterResult | None
    mode_used: str
    resolved_company_id: str | None
    days: int
    intent: str

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


__all__ = ["AgentState", "Message"]
