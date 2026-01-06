"""SSE streaming adapter for LangGraph agent execution."""

import json
import logging
import time
from typing import AsyncGenerator, Any
from datetime import datetime, date

from backend.agent.nodes.state import AgentState
from backend.agent.nodes.support.audit import AgentAuditLogger
from backend.agent.handlers.common import enrich_raw_data
from backend.agent.nodes.graph import build_agent_graph
from backend.agent.nodes.support.session import get_checkpointer, build_thread_config

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
    "fetch": "Fetching data...",
    "answer": "Generating answer...",
    "followup": "Generating suggestions...",
}


def serialize_for_json(obj: Any) -> Any:
    """Recursively serialize objects for JSON."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    return str(obj)


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(serialize_for_json(data))}\n\n"


async def stream_agent(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream agent execution as SSE using LangGraph's astream_events."""
    start = time.time()
    graph = build_agent_graph(checkpointer=get_checkpointer() if session_id else None)

    state: AgentState = {
        "question": question, "mode": mode, "company_id": company_id,
        "session_id": session_id, "user_id": user_id,
        "sources": [], "steps": [], "raw_data": {}, "follow_up_suggestions": [],
    }

    final, answer, phase = {}, "", None

    try:
        async for e in graph.astream_events(state, config=build_thread_config(session_id), version="v2"):
            typ, name = e.get("event"), e.get("name", "")

            # Node started → emit status
            if typ == "on_chain_start" and name in NODE_MESSAGES and name != phase:
                phase = name
                yield format_sse(StreamEvent.STATUS, {"message": NODE_MESSAGES[name]})
                if name == "answer":
                    yield format_sse(StreamEvent.ANSWER_START, {})

            # LLM token → emit chunk (only in answer phase)
            elif typ == "on_chat_model_stream" and phase == "answer":
                if chunk := e.get("data", {}).get("chunk"):
                    if content := getattr(chunk, "content", ""):
                        answer += content
                        yield format_sse(StreamEvent.ANSWER_CHUNK, {"chunk": content})

            # Graph done → capture final state
            elif typ == "on_chain_end" and name == "LangGraph":
                if out := e.get("data", {}).get("output"):
                    final = out if isinstance(out, dict) else {}

        # Finalize
        answer = final.get("answer", answer)
        if not answer:
            yield format_sse(StreamEvent.ANSWER_START, {})
        yield format_sse(StreamEvent.ANSWER_END, {"answer": answer})

        if sug := final.get("follow_up_suggestions", []):
            yield format_sse(StreamEvent.FOLLOWUP, {"suggestions": sug})

        AgentAuditLogger().log_query(
            question=question, mode_used=final.get("mode_used", "unknown"),
            company_id=final.get("resolved_company_id"),
            latency_ms=int((time.time() - start) * 1000),
            source_count=len(final.get("sources", [])),
            user_id=user_id, session_id=session_id,
        )

        yield format_sse(StreamEvent.DONE, {
            "answer": answer,
            "raw_data": enrich_raw_data(final.get("raw_data", {})),
            "follow_up_suggestions": final.get("follow_up_suggestions", []),
        })

    except Exception as ex:
        logger.error(f"[Stream] {ex}")
        yield format_sse(StreamEvent.ERROR, {"message": str(ex), "latency_ms": int((time.time() - start) * 1000)})


__all__ = ["stream_agent", "StreamEvent", "format_sse", "serialize_for_json"]
