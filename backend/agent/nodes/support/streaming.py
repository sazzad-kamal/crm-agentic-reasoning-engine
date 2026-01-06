"""
Streaming support for the agent graph.

Provides async generator that yields Server-Sent Events (SSE)
as the agent progresses through nodes.
"""

import json
import logging
import time
from typing import AsyncGenerator, Any
from datetime import datetime, date

from backend.agent.nodes.state import AgentState
from backend.agent.core.config import get_config
from backend.agent.nodes.support.audit import AgentAuditLogger
from backend.agent.nodes.support.formatters import (
    format_company_section,
    format_activities_section,
    format_history_section,
    format_pipeline_section,
    format_renewals_section,
    format_contacts_section,
    format_groups_section,
    format_attachments_section,
    format_docs_section,
    format_account_context_section,
    format_conversation_history_section,
)
from backend.agent.llm.helpers import stream_answer_chain, generate_follow_up_suggestions
from backend.agent.handlers.common import enrich_raw_data
from backend.agent.nodes.routing import route_node
from backend.agent.nodes.fetching import fetch_node

logger = logging.getLogger(__name__)


# =============================================================================
# JSON Serialization Helper
# =============================================================================


def serialize_for_json(obj: Any) -> Any:
    """Recursively serialize objects for JSON, handling Pydantic models and special types."""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif hasattr(obj, "model_dump"):
        # Pydantic v2 model
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        # Pydantic v1 model
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    else:
        # Fallback to string representation
        return str(obj)


# =============================================================================
# Event Types
# =============================================================================


class StreamEvent:
    """Event types for SSE streaming."""

    STATUS = "status"  # Progress update (e.g., "Routing question...")
    ANSWER_START = "answer_start"  # Answer generation starting
    ANSWER_CHUNK = "answer_chunk"  # Answer token/chunk
    ANSWER_END = "answer_end"  # Answer complete
    FOLLOWUP = "followup"  # Follow-up suggestions
    DONE = "done"  # Stream complete with final response
    ERROR = "error"  # Error occurred


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format data as Server-Sent Event."""
    # Serialize all data to ensure JSON compatibility
    serialized_data = serialize_for_json(data)
    return f"event: {event}\ndata: {json.dumps(serialized_data)}\n\n"


# =============================================================================
# Node to Status Mapping
# =============================================================================

NODE_MESSAGES = {
    "route": "Understanding your question...",
    "fetch": "Fetching data and documentation...",
    "answer": "Generating answer...",
    "followup": "Generating suggestions...",
}


# =============================================================================
# Streaming Runner
# =============================================================================


async def stream_agent(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream agent execution as Server-Sent Events.

    Yields SSE-formatted strings as the agent progresses.

    Events:
        - status: Progress messages (e.g., "Understanding your question...")
        - answer_start: Answer generation begins
        - answer_chunk: Incremental answer text (token-by-token streaming)
        - answer_end: Full answer available
        - followup: Follow-up suggestions
        - done: Final complete response
        - error: Error occurred

    Args:
        question: User's question
        mode: Query mode
        company_id: Optional pre-specified company
        session_id: Optional session ID
        user_id: Optional user ID

    Yields:
        SSE-formatted event strings
    """
    start_time = time.time()

    # Initialize state
    initial_state: AgentState = {
        "question": question,
        "mode": mode,
        "company_id": company_id,
        "session_id": session_id,
        "user_id": user_id,
        "sources": [],
        "steps": [],
        "raw_data": {},
        "follow_up_suggestions": [],
    }

    # Track accumulated state for final response
    final_state = initial_state.copy()

    try:
        # Phase 1: Route the question
        yield format_sse(StreamEvent.STATUS, {"message": NODE_MESSAGES["route"]})

        route_result = route_node(final_state)
        for key, value in route_result.items():
            final_state[key] = value

        # Phase 2: Fetch data
        yield format_sse(StreamEvent.STATUS, {"message": NODE_MESSAGES["fetch"]})

        fetch_result = fetch_node(final_state)
        for key, value in fetch_result.items():
            final_state[key] = value

        # Phase 3: Stream the answer
        yield format_sse(StreamEvent.STATUS, {"message": NODE_MESSAGES["answer"]})
        yield format_sse(StreamEvent.ANSWER_START, {})

        answer_start_time = time.time()
        accumulated_answer = ""

        # Build context sections from fetched data
        company_data = final_state.get("company_data")

        # Check if company was not found
        if company_data and not company_data.get("found"):
            # For not-found case, use a simple message
            close_matches = company_data.get("close_matches", [])[:3]
            answer = (
                "I couldn't find an exact match for that company in the CRM. "
                "Could you clarify which company you're asking about?"
            )
            if close_matches:
                answer += " Here are some similar companies I found: "
                answer += ", ".join(m.get("name", "") for m in close_matches)
            yield format_sse(StreamEvent.ANSWER_CHUNK, {"chunk": answer})
            accumulated_answer = answer
        else:
            # Stream answer tokens from LLM
            async for token in stream_answer_chain(
                question=question,
                conversation_history_section=format_conversation_history_section(final_state.get("messages", [])),
                company_section=format_company_section(company_data),
                contacts_section=format_contacts_section(final_state.get("contacts_data")),
                activities_section=format_activities_section(final_state.get("activities_data")),
                history_section=format_history_section(final_state.get("history_data")),
                pipeline_section=format_pipeline_section(final_state.get("pipeline_data")),
                renewals_section=format_renewals_section(final_state.get("renewals_data")),
                groups_section=format_groups_section(final_state.get("groups_data")),
                attachments_section=format_attachments_section(final_state.get("attachments_data")),
                docs_section=format_docs_section(final_state.get("docs_answer", "")),
                account_context_section=format_account_context_section(final_state.get("account_context_answer", "")),
            ):
                accumulated_answer += token
                yield format_sse(StreamEvent.ANSWER_CHUNK, {"chunk": token})

        final_state["answer"] = accumulated_answer
        yield format_sse(StreamEvent.ANSWER_END, {"answer": accumulated_answer})

        # Phase 4: Generate follow-up suggestions
        yield format_sse(StreamEvent.STATUS, {"message": NODE_MESSAGES["followup"]})

        config = get_config()
        if config.enable_follow_up_suggestions:
            company_name = None
            if company_data and company_data.get("found"):
                company_name = company_data.get("company", {}).get("name")

            raw_data = final_state.get("raw_data", {})
            available_data = {
                "contacts": len(raw_data.get("contacts", [])) if isinstance(raw_data, dict) else 0,
                "activities": len(raw_data.get("activities", [])) if isinstance(raw_data, dict) else 0,
                "opportunities": len(raw_data.get("opportunities", [])) if isinstance(raw_data, dict) else 0,
                "history": len(raw_data.get("history", [])) if isinstance(raw_data, dict) else 0,
                "renewals": len(raw_data.get("renewals", [])) if isinstance(raw_data, dict) else 0,
            }

            suggestions = generate_follow_up_suggestions(
                question=question,
                mode=final_state.get("mode_used", "auto"),
                company_id=final_state.get("resolved_company_id"),
                company_name=company_name,
                available_data=available_data,
            )
            final_state["follow_up_suggestions"] = suggestions

            if suggestions:
                yield format_sse(StreamEvent.FOLLOWUP, {"suggestions": suggestions})

        # Calculate total latency for audit logging
        latency_ms = int((time.time() - start_time) * 1000)

        # Audit logging
        audit = AgentAuditLogger()
        audit.log_query(
            question=question,
            mode_used=final_state.get("mode_used", "unknown"),
            company_id=final_state.get("resolved_company_id"),
            latency_ms=latency_ms,
            source_count=len(final_state.get("sources", [])),
            user_id=user_id,
            session_id=session_id,
        )

        # Emit final done event with complete response
        raw_data = enrich_raw_data(final_state.get("raw_data", {}))
        yield format_sse(
            StreamEvent.DONE,
            {
                "answer": final_state.get("answer", ""),
                "raw_data": raw_data,
                "follow_up_suggestions": final_state.get("follow_up_suggestions", []),
            },
        )

    except Exception as e:
        logger.error(f"[Stream] Error: {e}")
        yield format_sse(
            StreamEvent.ERROR,
            {
                "message": str(e),
                "latency_ms": int((time.time() - start_time) * 1000),
            },
        )


__all__ = [
    "stream_agent",
    "StreamEvent",
    "format_sse",
    "serialize_for_json",
]
