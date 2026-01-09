"""SSE streaming adapter for LangGraph agent execution."""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi.encoders import jsonable_encoder

from backend.agent.audit import AgentAuditLogger
from backend.agent.core.state import AgentState
from backend.agent.fetch.tools.common import enrich_raw_data
from backend.agent.graph import agent_graph, build_thread_config

logger = logging.getLogger(__name__)


class StreamEvent:
    STATUS = "status"
    ANSWER_START = "answer_start"
    ANSWER_CHUNK = "answer_chunk"
    ANSWER_END = "answer_end"
    FOLLOWUP = "followup"
    DONE = "done"
    ERROR = "error"


NODE_MESSAGES = {
    "route": "Understanding your question...",
    "fetch_crm": "Fetching CRM data...",
    "fetch_account": "Searching account context...",
    "answer": "Generating answer...",
    "followup": "Generating suggestions...",
}


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(jsonable_encoder(data))}\n\n"


async def stream_agent(question: str, session_id: str | None = None) -> AsyncGenerator[str, None]:
    """Stream agent execution as SSE events."""
    start = time.time()
    config = build_thread_config(session_id)

    # Don't pass messages - LangGraph checkpointer provides them from previous runs
    state: AgentState = {
        "question": question,
        "sources": [], "steps": [], "raw_data": {}, "follow_up_suggestions": [],
    }

    final: dict[str, Any] = {}
    phase: str | None = None

    try:
        async for e in agent_graph.astream_events(state, config=config, version="v2"):
            typ, name = e.get("event"), e.get("name", "")

            match (typ, name):
                case ("on_chain_start", n) if n in NODE_MESSAGES and n != phase:
                    phase = n
                    yield _format_sse(StreamEvent.STATUS, {"message": NODE_MESSAGES[n]})
                    if n == "answer":
                        yield _format_sse(StreamEvent.ANSWER_START, {})

                case ("on_chat_model_stream", _) if phase == "answer":
                    if content := getattr(e.get("data", {}).get("chunk"), "content", ""):
                        yield _format_sse(StreamEvent.ANSWER_CHUNK, {"chunk": content})

                case ("on_chain_end", "LangGraph"):
                    final = e.get("data", {}).get("output") or {}

        yield _format_sse(StreamEvent.ANSWER_END, {"answer": final.get("answer", "")})

        if sug := final.get("follow_up_suggestions"):
            yield _format_sse(StreamEvent.FOLLOWUP, {"suggestions": sug})

        AgentAuditLogger().log_query(
            question=question,
            company_id=final.get("resolved_company_id"),
            latency_ms=int((time.time() - start) * 1000),
            source_count=len(final.get("sources", [])),
            session_id=session_id,
        )

        # Messages are persisted automatically by LangGraph checkpointer
        # (answer_node updates state.messages, checkpointer saves it)

        yield _format_sse(StreamEvent.DONE, {
            "answer": final.get("answer", ""),
            "raw_data": enrich_raw_data(final.get("raw_data", {})),
            "follow_up_suggestions": final.get("follow_up_suggestions", []),
        })

    except Exception as ex:
        logger.error(f"[Stream] {ex}")
        yield _format_sse(StreamEvent.ERROR, {"message": str(ex), "latency_ms": int((time.time() - start) * 1000)})


__all__ = ["stream_agent", "StreamEvent"]
