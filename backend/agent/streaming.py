"""SSE streaming adapter for LangGraph agent execution."""

import json
import logging
from collections.abc import AsyncGenerator

from fastapi.encoders import jsonable_encoder

from backend.agent.core.state import AgentState
from backend.agent.graph import ANSWER_NODE, agent_graph, build_thread_config

logger = logging.getLogger(__name__)


class StreamEvent:
    ANSWER_CHUNK = "answer_chunk"
    ANSWER_END = "answer_end"
    DONE = "done"
    ERROR = "error"


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(jsonable_encoder(data))}\n\n"


async def stream_agent(question: str, session_id: str | None = None) -> AsyncGenerator[str, None]:
    """Stream agent execution as SSE events."""
    config = build_thread_config(session_id)
    state: AgentState = {"question": question}
    in_answer_node = False

    try:
        async for e in agent_graph.astream_events(state, config=config, version="v2"):
            typ, name = e.get("event"), e.get("name", "")

            if typ == "on_chain_start" and name == ANSWER_NODE:
                in_answer_node = True

            elif typ == "on_chat_model_stream" and in_answer_node:
                if content := getattr(e.get("data", {}).get("chunk"), "content", ""):
                    yield _format_sse(StreamEvent.ANSWER_CHUNK, {"chunk": content})

            elif typ == "on_chain_end" and name == "LangGraph":
                final = e.get("data", {}).get("output") or {}
                yield _format_sse(StreamEvent.ANSWER_END, {"answer": final.get("answer", "")})
                yield _format_sse(StreamEvent.DONE, {
                    "raw_data": final.get("raw_data", {}),
                    "follow_up_suggestions": final.get("follow_up_suggestions", []),
                })

    except Exception as ex:
        logger.error("[Stream] %s", ex)
        yield _format_sse(StreamEvent.ERROR, {"message": str(ex)})


__all__ = ["stream_agent"]
