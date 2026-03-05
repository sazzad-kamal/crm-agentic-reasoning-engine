"""SSE streaming adapter for LangGraph agent execution."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from backend.agent.followup.tree.loader import get_starters
from backend.agent.graph import (
    ACTION_NODE,
    ANSWER_NODE,
    FETCH_NODE,
    FOLLOWUP_NODE,
    GRAPH_NAME,
    LangGraphEvent,
    agent_graph,
    build_thread_config,
)
from backend.agent.progress_queue import (
    clear_progress_queue,
    create_progress_queue,
    drain_progress_queue,
)
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


class StreamEvent:
    FETCH_START = "fetch_start"  # Sent when fetch begins with expected steps
    FETCH_PROGRESS = "fetch_progress"  # Sent as each step completes
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

    Uses a progress queue to emit real-time progress updates during fetch.
    The fetch node puts progress events on the queue, and we poll it here.
    """
    config = build_thread_config(session_id)
    state: AgentState = {"question": question}
    in_answer_node = False
    in_action_node = False
    in_fetch_node = False

    # Create progress queue for real-time updates from fetch node
    progress_queue = create_progress_queue()

    try:
        # Use anext with timeout to allow polling the progress queue
        event_stream = agent_graph.astream_events(state, config=config, version="v2")
        event_iter = event_stream.__aiter__()

        while True:
            # Poll for progress events while waiting for next LangGraph event
            try:
                # Short timeout to allow frequent queue polling during fetch
                timeout = 0.05 if in_fetch_node else 30.0
                e = await asyncio.wait_for(event_iter.__anext__(), timeout=timeout)
            except TimeoutError:
                # Timeout - check progress queue and continue waiting
                print(f"[Stream] Timeout waiting for event, in_fetch_node={in_fetch_node}", flush=True)
                if in_fetch_node:
                    progress_events = await drain_progress_queue(progress_queue)
                    for progress in progress_events:
                        yield _format_sse(StreamEvent.FETCH_PROGRESS, progress)
                continue
            except StopAsyncIteration:
                break

            event_type, name = e.get("event"), e.get("name", "")

            # Debug: log all node events
            if event_type in (LangGraphEvent.CHAIN_START, LangGraphEvent.GRAPH_END):
                print(f"[Stream] Event: {event_type} name={name}", flush=True)

            # Track fetch node state for queue polling
            if event_type == LangGraphEvent.CHAIN_START and name == FETCH_NODE:
                in_fetch_node = True

            if event_type == LangGraphEvent.CHAIN_START and name == ANSWER_NODE:
                in_answer_node = True

            elif event_type == LangGraphEvent.CHAIN_START and name == ACTION_NODE:
                in_action_node = True

            elif event_type == LangGraphEvent.LLM_STREAM and in_answer_node:
                if content := getattr(e.get("data", {}).get("chunk"), "content", ""):
                    yield _format_sse(StreamEvent.ANSWER_CHUNK, {"chunk": content})

            elif event_type == LangGraphEvent.LLM_STREAM and in_action_node:
                if content := getattr(e.get("data", {}).get("chunk"), "content", ""):
                    yield _format_sse(StreamEvent.ACTION_CHUNK, {"chunk": content})

            elif event_type == LangGraphEvent.GRAPH_END and name == ANSWER_NODE:
                in_answer_node = False

            elif event_type == LangGraphEvent.GRAPH_END and name == ACTION_NODE:
                in_action_node = False

            elif event_type == LangGraphEvent.GRAPH_END and name == GRAPH_NAME:
                final = e.get("data", {}).get("output") or {}
                done_payload = {
                    "answer": final.get("answer", ""),
                    "follow_up_suggestions": final.get("follow_up_suggestions", []),
                    "suggested_action": final.get("suggested_action"),
                    "sql_results": final.get("sql_results", {}),
                }
                yield _format_sse(StreamEvent.DONE, done_payload)

            # Per-section events (non-exclusive, checked independently)
            if event_type == LangGraphEvent.GRAPH_END and name in (FETCH_NODE, ACTION_NODE, FOLLOWUP_NODE):
                try:
                    output = e.get("data", {}).get("output") or {}
                    if name == FETCH_NODE:
                        in_fetch_node = False
                        # Drain any remaining progress events
                        progress_events = await drain_progress_queue(progress_queue)
                        for progress in progress_events:
                            yield _format_sse(StreamEvent.FETCH_PROGRESS, progress)

                        sql_results = output.get("sql_results", {})
                        yield _format_sse(StreamEvent.DATA_READY, {
                            "sql_results": sql_results,
                        })
                        # No data → action+followup nodes skipped; resolve skeletons immediately
                        if not sql_results.get("data"):
                            yield _format_sse(StreamEvent.ACTION_READY, {"suggested_action": None})
                            yield _format_sse(StreamEvent.FOLLOWUP_READY, {
                                "follow_up_suggestions": get_starters(),
                            })
                    elif name == ACTION_NODE:
                        yield _format_sse(StreamEvent.ACTION_READY, {
                            "suggested_action": output.get("suggested_action"),
                        })
                    elif name == FOLLOWUP_NODE:
                        yield _format_sse(StreamEvent.FOLLOWUP_READY, {
                            "follow_up_suggestions": output.get("follow_up_suggestions", []),
                        })
                except Exception as section_err:
                    logger.warning("[Stream] Section event %s failed: %s", name, section_err)

    except Exception as ex:
        logger.error("[Stream] %s", ex)
        yield _format_sse(StreamEvent.ERROR, {"message": str(ex)})
    finally:
        clear_progress_queue()


__all__ = ["stream_agent"]
