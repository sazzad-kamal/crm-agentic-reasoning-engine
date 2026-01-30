"""LangGraph agent orchestration: Fetch → Answer → [Action, Followup]."""

import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.agent.action.node import action_node
from backend.agent.answer.node import answer_node
from backend.agent.fetch import fetch_node
from backend.agent.followup.node import followup_node
from backend.agent.state import AgentState


# LangGraph event constants (not exported by langgraph package)
class LangGraphEvent:
    CHAIN_START = "on_chain_start"
    GRAPH_END = "on_chain_end"
    LLM_STREAM = "on_chat_model_stream"

GRAPH_NAME = "LangGraph"  # Name used for whole graph in events

# Node names
FETCH_NODE = "fetch"
ANSWER_NODE = "answer"
ACTION_NODE = "action"
FOLLOWUP_NODE = "followup"

def build_thread_config(session_id: str | None) -> RunnableConfig:
    """Build LangGraph config with thread_id for checkpointing."""
    return {"configurable": {"thread_id": session_id or str(uuid.uuid4())}}


def _route_after_answer(state: AgentState) -> list[str] | str:
    """Route after answer: skip action+followup when no data was fetched."""
    sql_results = state.get("sql_results", {})
    if sql_results.get("data"):
        return [ACTION_NODE, FOLLOWUP_NODE]
    return END


def _build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node(FETCH_NODE, fetch_node)
    graph.add_node(ANSWER_NODE, answer_node)
    graph.add_node(ACTION_NODE, action_node)
    graph.add_node(FOLLOWUP_NODE, followup_node)

    # Entry point
    graph.set_entry_point(FETCH_NODE)

    # Flow: fetch → answer → [action, followup] → END
    # When no data: fetch → answer → END (skip action+followup)
    graph.add_edge(FETCH_NODE, ANSWER_NODE)
    graph.add_conditional_edges(
        ANSWER_NODE,
        _route_after_answer,
    )
    graph.add_edge(ACTION_NODE, END)
    graph.add_edge(FOLLOWUP_NODE, END)

    return graph.compile(checkpointer=MemorySaver())


agent_graph = _build_graph()

__all__ = ["agent_graph", "build_thread_config", "LangGraphEvent", "GRAPH_NAME", "FETCH_NODE", "ANSWER_NODE", "ACTION_NODE", "FOLLOWUP_NODE", "_route_after_answer"]
