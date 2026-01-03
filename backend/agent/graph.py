"""
LangGraph-based agent orchestration.

Implements a minimal 4-node graph workflow for answering CRM questions:

                    ┌─────────┐
                    │  Route  │
                    └────┬────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   Fetch     │  (Parallel: CRM + Docs + Account)
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   Answer    │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │  Follow-up  │
                  └──────┬──────┘
                         │
                         ▼
                     ┌───────┐
                     │  END  │
                     └───────┘

Usage:
    from backend.agent.graph import agent_graph, run_agent

    result = run_agent("What's going on with Acme Manufacturing?")
"""

import logging
import time
from typing import Any

from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.agent.state import AgentState, Message
from backend.agent.nodes.routing import route_node
from backend.agent.nodes.fetching import fetch_node
from backend.agent.nodes.generation import answer_node, followup_node
from backend.agent.audit import AgentAuditLogger

# Import from extracted modules
from backend.agent.cache import make_cache_key, get_cached_result, set_cached_result, clear_query_cache
from backend.agent.conversation import get_checkpointer, get_session_state, get_session_messages, build_thread_config


logger = logging.getLogger(__name__)


# =============================================================================
# Graph Construction
# =============================================================================


def build_agent_graph(checkpointer: Any = None) -> Any:
    """
    Build the LangGraph workflow.

    Simplified 4-node architecture:
      Route → Fetch (parallel) → Answer → Followup

    Args:
        checkpointer: Optional LangGraph checkpointer for conversation persistence.
                     If None, uses the global MemorySaver.

    Returns compiled graph ready for execution.
    """
    graph = StateGraph(AgentState)

    # Add nodes (simplified 4-node architecture)
    graph.add_node("route", route_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("answer", answer_node)
    graph.add_node("followup", followup_node)

    # Set entry point
    graph.set_entry_point("route")

    # Linear flow: route → fetch → answer → followup → END
    graph.add_edge("route", "fetch")
    graph.add_edge("fetch", "answer")
    graph.add_edge("answer", "followup")
    graph.add_edge("followup", END)

    return graph.compile(checkpointer=checkpointer or get_checkpointer())


# Compile the graph once at module load
agent_graph = build_agent_graph()


# =============================================================================
# Transient Errors for Retry
# =============================================================================


class TransientAgentError(Exception):
    """Transient error that should trigger a retry."""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(TransientAgentError),
    reraise=True,
)
def _execute_graph(initial_state: AgentState, config: dict) -> dict:
    """Execute the graph with retry logic for transient failures."""
    try:
        return agent_graph.invoke(initial_state, config=config)
    except ConnectionError as e:
        logger.warning(f"[Agent] Transient connection error, will retry: {e}")
        raise TransientAgentError(str(e)) from e
    except TimeoutError as e:
        logger.warning(f"[Agent] Transient timeout error, will retry: {e}")
        raise TransientAgentError(str(e)) from e


# =============================================================================
# Runner Function
# =============================================================================


def run_agent(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Run the agent graph and return formatted response.

    Args:
        question: The user's question
        mode: Mode override ("auto", "docs", "data", "data+docs")
        company_id: Pre-specified company ID
        session_id: Optional session ID (used as thread_id for checkpointing)
        user_id: Optional user ID
        use_cache: Whether to use query cache (default True)

    Returns:
        Dict matching ChatResponse schema
    """
    start_time = time.time()

    # Check query cache first (only for non-session queries)
    cache_key = None
    if use_cache and not session_id:
        cache_key = make_cache_key(question, mode, company_id)
        cached = get_cached_result(cache_key)
        if cached:
            logger.info("[Agent] Cache hit, returning cached result")
            cached_copy = cached.copy()
            cached_copy["meta"] = cached["meta"].copy()
            cached_copy["meta"]["latency_ms"] = int((time.time() - start_time) * 1000)
            cached_copy["meta"]["cached"] = True
            return cached_copy

    # Load conversation history from checkpoint
    messages: list[Message] = get_session_messages(session_id) if session_id else []

    # Initialize state
    initial_state: AgentState = {
        "question": question,
        "mode": mode,
        "company_id": company_id,
        "session_id": session_id,
        "user_id": user_id,
        "messages": messages,
        "sources": [],
        "steps": [],
        "raw_data": {},
        "follow_up_suggestions": [],
    }

    # Build config with thread_id
    config = build_thread_config(session_id)
    if session_id:
        logger.debug(f"[Agent] Using LangGraph checkpointing with thread_id={session_id}")

    # Run the graph
    logger.info(f"[Agent] Starting graph execution for: {question[:50]}...")

    try:
        final_state = _execute_graph(initial_state, config)
    except TransientAgentError as e:
        logger.error(f"[Agent] Graph execution failed after retries: {e}")
        return _build_error_response(str(e), start_time)
    except Exception as e:
        logger.error(f"[Agent] Graph execution failed: {e}")
        return _build_error_response(str(e), start_time)

    latency_ms = int((time.time() - start_time) * 1000)

    # Audit logging
    AgentAuditLogger().log_query(
        question=question,
        mode_used=final_state.get("mode_used", "unknown"),
        company_id=final_state.get("resolved_company_id"),
        latency_ms=latency_ms,
        source_count=len(final_state.get("sources", [])),
        user_id=user_id,
        session_id=session_id,
    )

    logger.info(f"[Agent] Complete in {latency_ms}ms")

    # Build response
    result = {
        "answer": final_state.get("answer", ""),
        "sources": [
            s.model_dump() if hasattr(s, "model_dump") else s
            for s in final_state.get("sources", [])
        ],
        "steps": final_state.get("steps", []),
        "raw_data": final_state.get("raw_data", {}),
        "follow_up_suggestions": final_state.get("follow_up_suggestions", []),
        "meta": {
            "mode_used": final_state.get("mode_used", "unknown"),
            "latency_ms": latency_ms,
            "company_id": final_state.get("resolved_company_id"),
            "intent": final_state.get("intent", "general"),
            "days": final_state.get("days", 90),
            "router_latency_ms": final_state.get("router_latency_ms"),
            "fetch_latency_ms": final_state.get("fetch_latency_ms"),
            "answer_latency_ms": final_state.get("answer_latency_ms"),
            "followup_latency_ms": final_state.get("followup_latency_ms"),
        },
    }

    # Cache result for non-session queries
    if cache_key:
        set_cached_result(cache_key, result)

    return result


def _build_error_response(error: str, start_time: float) -> dict[str, Any]:
    """Build error response with consistent structure."""
    return {
        "answer": f"I'm sorry, I encountered an error: {error}",
        "sources": [],
        "steps": [{"id": "error", "label": f"Error: {error[:50]}", "status": "error"}],
        "raw_data": {},
        "follow_up_suggestions": [],
        "meta": {
            "mode_used": "error",
            "latency_ms": int((time.time() - start_time) * 1000),
            "router_latency_ms": None,
            "fetch_latency_ms": None,
            "answer_latency_ms": None,
            "followup_latency_ms": None,
        },
    }


# =============================================================================
# Backwards Compatibility
# =============================================================================


def answer_question(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Backwards-compatible wrapper for run_agent."""
    return run_agent(
        question=question,
        mode=mode,
        company_id=company_id,
        session_id=session_id,
        user_id=user_id,
    )


def get_graph_mermaid() -> str:
    """Get Mermaid diagram of the graph."""
    return """
graph TD
    START((Start)) --> route[Route]
    route --> fetch[Fetch<br/>CRM + Docs + Account]
    fetch --> answer[Synthesize Answer]
    answer --> followup[Generate Follow-ups]
    followup --> END((End))

    style route fill:#e1f5fe
    style fetch fill:#fff3e0
    style answer fill:#e8f5e9
    style followup fill:#fce4ec
"""


__all__ = [
    "agent_graph",
    "run_agent",
    "answer_question",
    "build_agent_graph",
    "get_graph_mermaid",
    # Re-export from conversation module
    "get_checkpointer",
    "get_session_state",
    # Re-export from cache module
    "clear_query_cache",
    # Errors
    "TransientAgentError",
]
