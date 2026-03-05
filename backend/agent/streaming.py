"""SSE streaming adapter for LangGraph agent execution."""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from backend.agent.followup.tree.loader import get_starters
from backend.agent.graph import (
    agent_graph,
    build_thread_config,
)
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


class StreamEvent:
    FETCH_START = "fetch_start"
    FETCH_PROGRESS = "fetch_progress"
    ANSWER_CHUNK = "answer_chunk"
    ACTION_CHUNK = "action_chunk"
    DATA_READY = "data_ready"
    ACTION_READY = "action_ready"
    FOLLOWUP_READY = "followup_ready"
    DONE = "done"
    ERROR = "error"


def _format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream_agent(question: str, session_id: str | None = None) -> AsyncGenerator[str, None]:  # pragma: no cover
    """Stream agent execution as SSE events.

    Uses astream for node-level updates (more reliable than astream_events on Railway).
    """
    config = build_thread_config(session_id)
    state: AgentState = {"question": question}

    print(f"[Stream] Starting graph for: {question[:50]}...", flush=True)

    final_state: dict[str, Any] = {}

    try:
        # Use astream - yields state updates at each node boundary
        async for chunk in agent_graph.astream(state, config=config, stream_mode="updates"):
            print(f"[Stream] Node update: {list(chunk.keys())}", flush=True)

            for node_name, node_output in chunk.items():
                if node_name == "fetch":
                    sql_results = node_output.get("sql_results", {})
                    final_state["sql_results"] = sql_results
                    yield _format_sse(StreamEvent.DATA_READY, {"sql_results": sql_results})

                    # If no data, emit early action/followup
                    if not sql_results.get("data"):
                        yield _format_sse(StreamEvent.ACTION_READY, {"suggested_action": None})
                        yield _format_sse(StreamEvent.FOLLOWUP_READY, {
                            "follow_up_suggestions": get_starters(),
                        })

                elif node_name == "answer":
                    answer = node_output.get("answer", "")
                    final_state["answer"] = answer
                    # Stream the answer as chunks (simulate streaming)
                    # For now, send as single chunk since we don't have true token streaming
                    yield _format_sse(StreamEvent.ANSWER_CHUNK, {"chunk": answer})

                elif node_name == "action":
                    action = node_output.get("suggested_action")
                    final_state["suggested_action"] = action
                    yield _format_sse(StreamEvent.ACTION_READY, {"suggested_action": action})

                elif node_name == "followup":
                    followups = node_output.get("follow_up_suggestions", [])
                    final_state["follow_up_suggestions"] = followups
                    yield _format_sse(StreamEvent.FOLLOWUP_READY, {"follow_up_suggestions": followups})

        print("[Stream] Graph completed", flush=True)

        # Emit done event with final state
        yield _format_sse(StreamEvent.DONE, {
            "answer": final_state.get("answer", ""),
            "follow_up_suggestions": final_state.get("follow_up_suggestions", get_starters()),
            "suggested_action": final_state.get("suggested_action"),
            "sql_results": final_state.get("sql_results", {}),
        })

    except Exception as ex:
        logger.error("[Stream] %s", ex)
        print(f"[Stream] Error: {ex}", flush=True)
        yield _format_sse(StreamEvent.ERROR, {"message": str(ex)})


__all__ = ["stream_agent"]
