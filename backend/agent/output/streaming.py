"""
Streaming support for the agent graph.

Provides async generator that yields Server-Sent Events (SSE)
as the agent progresses through nodes.
"""

import json
import logging
import time
import uuid
from typing import AsyncGenerator, Any
from datetime import datetime, date

from backend.agent.graph import agent_graph
from backend.agent.core.state import AgentState
from backend.agent.output.audit import AgentAuditLogger

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
    STEP = "step"  # Step completed
    SOURCES = "sources"  # Sources discovered
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
    "data": "Fetching CRM data...",
    "docs": "Searching documentation...",
    "skip_data": "Skipping data (docs-only query)...",
    "skip_docs": "Skipping docs (data-only query)...",
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
        - status: Progress messages (e.g., "Routing question...")
        - step: Step completion with status
        - sources: Sources discovered during data/docs fetch
        - answer_start: Answer generation begins
        - answer_chunk: Incremental answer text (if LLM supports streaming)
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
    seen_steps = set()
    answer_started = False
    accumulated_answer = ""

    # Build config with thread_id for LangGraph checkpointing
    thread_id = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Stream events from the graph
        async for event in agent_graph.astream_events(initial_state, config=config, version="v2"):
            event_type = event.get("event")
            event_name = event.get("name")
            event_data = event.get("data", {})

            # Node starting
            if event_type == "on_chain_start" and event_name in NODE_MESSAGES:
                yield format_sse(
                    StreamEvent.STATUS,
                    {
                        "node": event_name,
                        "message": NODE_MESSAGES[event_name],
                    },
                )

            # LLM token streaming - emit tokens as they arrive
            elif event_type == "on_llm_stream":
                chunk = event_data.get("chunk")
                if chunk:
                    # Extract token content from AIMessageChunk
                    token = ""
                    if hasattr(chunk, "content"):
                        token = chunk.content
                    elif isinstance(chunk, dict):
                        token = chunk.get("content", "")

                    if token:
                        # Emit answer_start on first token
                        if not answer_started:
                            answer_started = True
                            yield format_sse(StreamEvent.ANSWER_START, {})

                        accumulated_answer += token
                        yield format_sse(StreamEvent.ANSWER_CHUNK, {"chunk": token})

            # Node completed - extract output
            elif event_type == "on_chain_end" and event_name in NODE_MESSAGES:
                output = event_data.get("output", {})

                # Merge output into final state
                if isinstance(output, dict):
                    for key, value in output.items():
                        if key == "sources" and isinstance(value, list):
                            # Append sources
                            if "sources" not in final_state:
                                final_state["sources"] = []
                            final_state["sources"].extend(value)
                        elif key == "steps" and isinstance(value, list):
                            # Append steps
                            if "steps" not in final_state:
                                final_state["steps"] = []
                            final_state["steps"].extend(value)
                        else:
                            final_state[key] = value

                # Emit steps that are new
                steps = output.get("steps", [])
                for step in steps:
                    step_id = step.get("id")
                    if step_id and step_id not in seen_steps:
                        seen_steps.add(step_id)
                        yield format_sse(StreamEvent.STEP, step)

                # Emit sources if any
                sources = output.get("sources", [])
                if sources:
                    yield format_sse(
                        StreamEvent.SOURCES,
                        {
                            "sources": sources  # format_sse handles serialization
                        },
                    )

                # Special handling for answer node
                if event_name == "answer":
                    # Use accumulated_answer if we streamed tokens, otherwise use output
                    answer = accumulated_answer if accumulated_answer else output.get("answer", "")
                    if answer:
                        # If we didn't stream (mock mode), emit the full answer
                        if not answer_started:
                            yield format_sse(StreamEvent.ANSWER_START, {})
                            yield format_sse(StreamEvent.ANSWER_CHUNK, {"chunk": answer})
                        # Always emit answer_end with the complete answer
                        yield format_sse(StreamEvent.ANSWER_END, {"answer": answer})
                        # Update final_state with the answer
                        final_state["answer"] = answer

                # Follow-up suggestions
                if event_name == "followup":
                    suggestions = output.get("follow_up_suggestions", [])
                    if suggestions:
                        yield format_sse(StreamEvent.FOLLOWUP, {"suggestions": suggestions})

        # Calculate latency
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
        yield format_sse(
            StreamEvent.DONE,
            {
                "answer": final_state.get("answer", ""),
                "sources": final_state.get("sources", []),  # format_sse handles serialization
                "steps": final_state.get("steps", []),
                "raw_data": final_state.get("raw_data", {}),
                "follow_up_suggestions": final_state.get("follow_up_suggestions", []),
                "meta": {
                    "mode_used": final_state.get("mode_used", "unknown"),
                    "latency_ms": latency_ms,
                    "company_id": final_state.get("resolved_company_id"),
                    "intent": final_state.get("intent", "general"),
                    "days": final_state.get("days", 90),
                },
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
