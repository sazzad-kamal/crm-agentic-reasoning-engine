"""LangGraph agent orchestration: Route → fetch_sql → fetch_rag → Answer → Followup."""

import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.agent.answer.node import answer_node
from backend.agent.core.state import AgentState
from backend.agent.fetch.fetch_rag import fetch_rag_node
from backend.agent.fetch.fetch_sql import fetch_sql_node
from backend.agent.followup.node import followup_node
from backend.agent.route.node import route_node


# LangGraph event constants (not exported by langgraph package)
class LangGraphEvent:
    CHAIN_START = "on_chain_start"
    GRAPH_END = "on_chain_end"
    LLM_STREAM = "on_chat_model_stream"

GRAPH_NAME = "LangGraph"  # Name used for whole graph in events

# Node names
ROUTE_NODE = "route"
FETCH_SQL_NODE = "fetch_sql"
FETCH_RAG_NODE = "fetch_rag"
ANSWER_NODE = "answer"
FOLLOWUP_NODE = "followup"

def build_thread_config(session_id: str | None) -> dict:
    """Build LangGraph config with thread_id for checkpointing."""
    return {"configurable": {"thread_id": session_id or str(uuid.uuid4())}}


def _build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node(ROUTE_NODE, route_node)
    graph.add_node(FETCH_SQL_NODE, fetch_sql_node)
    graph.add_node(FETCH_RAG_NODE, fetch_rag_node)
    graph.add_node(ANSWER_NODE, answer_node)
    graph.add_node(FOLLOWUP_NODE, followup_node)

    # Entry point
    graph.set_entry_point(ROUTE_NODE)

    # Sequential flow: route → fetch_sql → fetch_rag → answer → followup
    graph.add_edge(ROUTE_NODE, FETCH_SQL_NODE)
    graph.add_edge(FETCH_SQL_NODE, FETCH_RAG_NODE)
    graph.add_edge(FETCH_RAG_NODE, ANSWER_NODE)
    graph.add_edge(ANSWER_NODE, FOLLOWUP_NODE)
    graph.add_edge(FOLLOWUP_NODE, END)

    return graph.compile(checkpointer=MemorySaver())


agent_graph = _build_graph()

__all__ = ["agent_graph", "build_thread_config", "LangGraphEvent", "GRAPH_NAME", "ANSWER_NODE"]
