"""SSE streaming adapter for LangGraph agent execution."""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from backend.agent.graph import (
    ANSWER_NODE,
    GRAPH_NAME,
    LangGraphEvent,
    agent_graph,
    build_thread_config,
)
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


class StreamEvent:
    ANSWER_CHUNK = "answer_chunk"
    DONE = "done"
    ERROR = "error"


def _format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_agent(question: str, session_id: str | None = None) -> AsyncGenerator[str, None]:  # pragma: no cover
    """Stream agent execution as SSE events."""
    config = build_thread_config(session_id)
    state: AgentState = {"question": question}
    in_answer_node = False

    try:
        async for e in agent_graph.astream_events(state, config=config, version="v2"):
            event_type, name = e.get("event"), e.get("name", "")

            if event_type == LangGraphEvent.CHAIN_START and name == ANSWER_NODE:
                in_answer_node = True

            elif event_type == LangGraphEvent.LLM_STREAM and in_answer_node:
                if content := getattr(e.get("data", {}).get("chunk"), "content", ""):
                    yield _format_sse(StreamEvent.ANSWER_CHUNK, {"chunk": content})

            elif event_type == LangGraphEvent.GRAPH_END and name == ANSWER_NODE:
                in_answer_node = False

            elif event_type == LangGraphEvent.GRAPH_END and name == GRAPH_NAME:
                final = e.get("data", {}).get("output") or {}
                yield _format_sse(StreamEvent.DONE, {
                    "answer": final.get("answer", ""),
                    "sql_results": final.get("sql_results", {}),
                    "follow_up_suggestions": final.get("follow_up_suggestions", []),
                    "suggested_action": final.get("suggested_action"),
                })

    except Exception as ex:
        logger.error("[Stream] %s", ex)
        yield _format_sse(StreamEvent.ERROR, {"message": str(ex)})


__all__ = ["stream_agent"]
