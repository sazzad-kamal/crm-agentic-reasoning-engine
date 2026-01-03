"""
Agent state definition for LangGraph workflow.

This module defines the central state structure that flows through the graph.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict, Annotated
from operator import add

from backend.agent.schemas import Source, RouterResult, Step


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

    # Steps accumulated from all nodes (using reducer to append)
    steps: Annotated[list[dict[str, Any]], add]

    # Final outputs
    answer: str
    follow_up_suggestions: list[str]

    # Raw data for UI
    raw_data: dict[str, Any]

    # Error handling
    error: str | None


@dataclass
class AgentProgress:
    """Tracks progress through the agent pipeline."""

    steps: list[Step] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def add_step(self, step_id: str, label: str, status: str = "done") -> None:
        """Add a completed step."""
        self.steps.append(Step(id=step_id, label=label, status=status))
        logger.debug(f"Step: {step_id} - {label} [{status}]")

    def get_elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        return int((time.time() - self.start_time) * 1000)

    def to_list(self) -> list[dict]:
        """Convert steps to list of dicts."""
        return [step.model_dump() for step in self.steps]


__all__ = ["AgentState", "Message", "AgentProgress"]
