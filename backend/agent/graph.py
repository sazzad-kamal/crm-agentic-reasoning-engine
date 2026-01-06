"""
LangGraph-based agent orchestration.

Implements a 4-node graph: Route → Fetch → Answer → Followup.
"""

import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.agent.core.state import AgentState
from backend.agent.route.node import route_node
from backend.agent.fetch.node import fetch_node
from backend.agent.answer.node import answer_node
from backend.agent.followup.node import followup_node

_checkpointer = MemorySaver()


def build_thread_config(session_id: str | None) -> dict:
    """Build LangGraph config with thread_id for checkpointing."""
    return {"configurable": {"thread_id": session_id or str(uuid.uuid4())}}


def _build_graph():
    """Build the LangGraph workflow."""
    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("answer", answer_node)
    graph.add_node("followup", followup_node)
    graph.set_entry_point("route")
    graph.add_edge("route", "fetch")
    graph.add_edge("fetch", "answer")
    graph.add_edge("answer", "followup")
    graph.add_edge("followup", END)
    return graph.compile(checkpointer=_checkpointer)


agent_graph = _build_graph()

__all__ = ["agent_graph", "build_thread_config"]
