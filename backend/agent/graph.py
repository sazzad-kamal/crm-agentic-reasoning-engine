"""
LangGraph-based agent orchestration.

Implements a parallel fetch graph:
    Route ─┬─→ fetch_crm ──────┬─→ Answer → Followup
           ├─→ fetch_docs ─────┤
           └─→ fetch_account ──┘
"""

import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.agent.core.state import AgentState
from backend.agent.route.node import route_node
from backend.agent.fetch.fetch_crm import fetch_crm_node
from backend.agent.fetch.fetch_docs import fetch_docs_node
from backend.agent.fetch.fetch_account import fetch_account_node
from backend.agent.answer.node import answer_node
from backend.agent.followup.node import followup_node

_checkpointer = MemorySaver()


def build_thread_config(session_id: str | None) -> dict:
    """Build LangGraph config with thread_id for checkpointing."""
    return {"configurable": {"thread_id": session_id or str(uuid.uuid4())}}


def clear_thread(session_id: str | None) -> None:
    """Clear conversation history for a session by deleting checkpoint."""
    if not session_id:
        return
    # MemorySaver stores checkpoints keyed by (thread_id, checkpoint_ns, checkpoint_id)
    keys_to_delete = [k for k in _checkpointer.storage.keys() if k[0] == session_id]
    for key in keys_to_delete:
        del _checkpointer.storage[key]


def _build_graph():
    """Build the LangGraph workflow with parallel fetch nodes."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route", route_node)
    graph.add_node("fetch_crm", fetch_crm_node)
    graph.add_node("fetch_docs", fetch_docs_node)
    graph.add_node("fetch_account", fetch_account_node)
    graph.add_node("answer", answer_node)
    graph.add_node("followup", followup_node)

    # Entry point
    graph.set_entry_point("route")

    # Fan-out: route → all fetch nodes (parallel execution)
    graph.add_edge("route", "fetch_crm")
    graph.add_edge("route", "fetch_docs")
    graph.add_edge("route", "fetch_account")

    # Fan-in: all fetch nodes → answer (waits for all to complete)
    graph.add_edge("fetch_crm", "answer")
    graph.add_edge("fetch_docs", "answer")
    graph.add_edge("fetch_account", "answer")

    # Continue to followup and end
    graph.add_edge("answer", "followup")
    graph.add_edge("followup", END)

    return graph.compile(checkpointer=_checkpointer)


agent_graph = _build_graph()

__all__ = ["agent_graph", "build_thread_config", "clear_thread"]
