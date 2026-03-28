"""LangGraph multi-agent orchestration with Supervisor routing.

Architecture:
    Supervisor → (routes by intent)
        ├── data_query → Fetch → Answer → (loop if needs_more_data)
        ├── compare → Compare → Answer → [Action, Followup]
        ├── trend → Trend → Answer → [Action, Followup]
        ├── complex → Planner → Answer → [Action, Followup]
        ├── export → Export → Answer → [Action, Followup]
        ├── health → Health → Answer → [Action, Followup]
        ├── docs → RAG → Answer → [Action, Followup]  # LlamaIndex documentation search
        ├── graph → Graph → Answer → [Action, Followup]  # Neo4j multi-hop queries
        ├── clarify → Answer (asks for clarification)
        └── help → Answer (responds without SQL)

    After Answer (if complete) → [Action, Followup] → END
"""

import uuid
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from backend.agent.action.node import action_node
from backend.agent.answer.node import answer_node
from backend.agent.compare.node import compare_node
from backend.agent.export.node import export_node
from backend.agent.fetch import fetch_node
from backend.agent.followup.node import followup_node
from backend.agent.graph_rag.node import graph_node
from backend.agent.health.node import health_node
from backend.agent.planner.node import planner_node
from backend.agent.rag.node import rag_node
from backend.agent.state import AgentState
from backend.agent.supervisor import supervisor_node
from backend.agent.trend.node import trend_node


# LangGraph event constants (not exported by langgraph package)
class LangGraphEvent:
    CHAIN_START = "on_chain_start"
    GRAPH_END = "on_chain_end"
    LLM_STREAM = "on_chat_model_stream"

GRAPH_NAME = "LangGraph"  # Name used for whole graph in events

# Node names
SUPERVISOR_NODE = "supervisor"
FETCH_NODE = "fetch"
COMPARE_NODE = "compare"
TREND_NODE = "trend"
PLANNER_NODE = "planner"
EXPORT_NODE = "export"
HEALTH_NODE = "health"
RAG_NODE = "rag"
GRAPH_RAG_NODE = "graph"
ANSWER_NODE = "answer"
ACTION_NODE = "action"
FOLLOWUP_NODE = "followup"


def build_thread_config(session_id: str | None) -> RunnableConfig:
    """Build LangGraph config with thread_id for checkpointing."""
    return {"configurable": {"thread_id": session_id or str(uuid.uuid4())}}


def _route_after_supervisor(
    state: AgentState,
) -> Literal["fetch", "compare", "trend", "planner", "export", "health", "rag", "graph", "answer"]:
    """Route based on intent to the appropriate specialized agent."""
    intent = state.get("intent", "data_query")

    # Route to specialized agents
    intent_to_node = {
        "data_query": FETCH_NODE,
        "compare": COMPARE_NODE,
        "trend": TREND_NODE,
        "complex": PLANNER_NODE,
        "export": EXPORT_NODE,
        "health": HEALTH_NODE,
        "docs": RAG_NODE,
        "graph": GRAPH_RAG_NODE,
    }

    if intent in intent_to_node:
        return intent_to_node[intent]

    # clarify or help: go directly to answer (no SQL needed)
    return ANSWER_NODE


def _route_after_answer(state: AgentState) -> list[str] | str:
    """Route after answer: loop back to fetch, or continue to action/followup.

    Decision logic:
    1. If needs_more_data=True → loop back to Fetch (only for data_query intent)
    2. If intent was clarify/help → END (no action/followup)
    3. If data was fetched → [Action, Followup]
    4. Otherwise → END
    """
    intent = state.get("intent", "data_query")

    # Check for data refinement loop (only for data_query intent)
    if state.get("needs_more_data", False) and intent == "data_query":
        return FETCH_NODE

    # Non-data intents don't need action/followup
    if intent in ("clarify", "help"):
        return END

    # All data intents: include action/followup if we have results
    sql_results = state.get("sql_results", {})
    has_data = (
        sql_results.get("data") or
        sql_results.get("comparison") or
        sql_results.get("trend_analysis") or
        sql_results.get("aggregated") or
        sql_results.get("export") or
        sql_results.get("health_analysis") or
        sql_results.get("rag_answer") or  # RAG documentation results
        sql_results.get("graph_data")  # Neo4j graph results
    )

    if has_data:
        return [ACTION_NODE, FOLLOWUP_NODE]

    return END


def _build_graph() -> Any:
    """Build and compile the LangGraph multi-agent workflow with Supervisor routing."""
    graph = StateGraph(AgentState)

    # Add nodes - Supervisor
    graph.add_node(SUPERVISOR_NODE, supervisor_node)

    # Add nodes - Specialized agents
    graph.add_node(FETCH_NODE, fetch_node)
    graph.add_node(COMPARE_NODE, compare_node)
    graph.add_node(TREND_NODE, trend_node)
    graph.add_node(PLANNER_NODE, planner_node)
    graph.add_node(EXPORT_NODE, export_node)
    graph.add_node(HEALTH_NODE, health_node)
    graph.add_node(RAG_NODE, rag_node)
    graph.add_node(GRAPH_RAG_NODE, graph_node)

    # Add nodes - Response generation
    graph.add_node(ANSWER_NODE, answer_node)
    graph.add_node(ACTION_NODE, action_node)
    graph.add_node(FOLLOWUP_NODE, followup_node)

    # Entry point: Supervisor classifies intent
    graph.set_entry_point(SUPERVISOR_NODE)

    # Supervisor routes based on intent to specialized agents
    graph.add_conditional_edges(
        SUPERVISOR_NODE,
        _route_after_supervisor,
    )

    # All specialized agents flow to Answer
    graph.add_edge(FETCH_NODE, ANSWER_NODE)
    graph.add_edge(COMPARE_NODE, ANSWER_NODE)
    graph.add_edge(TREND_NODE, ANSWER_NODE)
    graph.add_edge(PLANNER_NODE, ANSWER_NODE)
    graph.add_edge(EXPORT_NODE, ANSWER_NODE)
    graph.add_edge(HEALTH_NODE, ANSWER_NODE)
    graph.add_edge(RAG_NODE, ANSWER_NODE)
    graph.add_edge(GRAPH_RAG_NODE, ANSWER_NODE)

    # Answer routes: loop back to Fetch OR continue to Action/Followup OR END
    graph.add_conditional_edges(
        ANSWER_NODE,
        _route_after_answer,
    )

    # Terminal edges
    graph.add_edge(ACTION_NODE, END)
    graph.add_edge(FOLLOWUP_NODE, END)

    return graph.compile(checkpointer=MemorySaver())


agent_graph = _build_graph()

__all__ = [
    "agent_graph",
    "build_thread_config",
    "LangGraphEvent",
    "GRAPH_NAME",
    "SUPERVISOR_NODE",
    "FETCH_NODE",
    "COMPARE_NODE",
    "TREND_NODE",
    "PLANNER_NODE",
    "EXPORT_NODE",
    "HEALTH_NODE",
    "RAG_NODE",
    "GRAPH_RAG_NODE",
    "ANSWER_NODE",
    "ACTION_NODE",
    "FOLLOWUP_NODE",
    "_route_after_supervisor",
    "_route_after_answer",
]
