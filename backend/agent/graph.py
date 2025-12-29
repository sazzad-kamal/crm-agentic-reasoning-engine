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

4 nodes total. The Fetch node always runs CRM data + Docs RAG in parallel,
with optional Account RAG for company-specific intents.

Usage:
    from backend.agent.graph import agent_graph, run_agent

    result = run_agent("What's going on with Acme Manufacturing?")
"""

import hashlib
import logging
import time
import uuid
from functools import lru_cache
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.agent.state import AgentState, Message
from backend.agent.nodes import (
    route_node,
    fetch_node,
    answer_node,
    followup_node,
)
from backend.agent.audit import AgentAuditLogger


logger = logging.getLogger(__name__)


# =============================================================================
# LangGraph Checkpointing
# =============================================================================

# Global checkpointer for conversation persistence
_checkpointer = MemorySaver()


# =============================================================================
# Query Cache
# =============================================================================

# Cache for repeated identical queries (max 128 entries)
_query_cache: dict[str, tuple[dict, float]] = {}
_CACHE_MAX_SIZE = 128
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _make_cache_key(question: str, mode: str, company_id: str | None) -> str:
    """Generate cache key from query parameters."""
    key_data = f"{question.lower().strip()}|{mode}|{company_id or ''}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _get_cached_result(cache_key: str) -> dict | None:
    """Get cached result if valid (not expired)."""
    if cache_key in _query_cache:
        result, timestamp = _query_cache[cache_key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            logger.debug(f"[Cache] Hit for key {cache_key}")
            return result
        else:
            # Expired, remove from cache
            del _query_cache[cache_key]
            logger.debug(f"[Cache] Expired key {cache_key}")
    return None


def _set_cached_result(cache_key: str, result: dict) -> None:
    """Store result in cache with timestamp."""
    # Evict oldest entries if cache is full
    if len(_query_cache) >= _CACHE_MAX_SIZE:
        # Remove oldest entry (first inserted)
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]
        logger.debug(f"[Cache] Evicted oldest key {oldest_key}")

    _query_cache[cache_key] = (result, time.time())
    logger.debug(f"[Cache] Stored key {cache_key}")


def clear_query_cache() -> None:
    """Clear the query cache (useful for testing)."""
    _query_cache.clear()
    logger.debug("[Cache] Cleared all entries")


# =============================================================================
# Graph Construction
# =============================================================================

def build_agent_graph(checkpointer=None):
    """
    Build the LangGraph workflow.

    Simplified 4-node architecture:
      Route → Fetch (parallel) → Answer → Followup

    The Fetch node always runs CRM data + Docs RAG in parallel,
    with optional Account RAG for company-specific intents.

    Args:
        checkpointer: Optional LangGraph checkpointer for conversation persistence.
                     If None, uses the global MemorySaver.

    Returns compiled graph ready for execution.
    """
    # Create graph with state schema
    graph = StateGraph(AgentState)

    # Add nodes (simplified 4-node architecture)
    graph.add_node("route", route_node)
    graph.add_node("fetch", fetch_node)  # Unified parallel fetch
    graph.add_node("answer", answer_node)
    graph.add_node("followup", followup_node)

    # Set entry point
    graph.set_entry_point("route")

    # Linear flow: route → fetch → answer → followup → END
    graph.add_edge("route", "fetch")
    graph.add_edge("fetch", "answer")
    graph.add_edge("answer", "followup")
    graph.add_edge("followup", END)

    # Compile with checkpointer for conversation persistence
    return graph.compile(checkpointer=checkpointer or _checkpointer)


# Compile the graph once at module load (with checkpointing enabled)
agent_graph = build_agent_graph()


# =============================================================================
# Transient Errors for Retry
# =============================================================================

class TransientAgentError(Exception):
    """Transient error that should trigger a retry."""
    pass


# =============================================================================
# Runner Function
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(TransientAgentError),
    reraise=True,
)
def _execute_graph(
    initial_state: AgentState,
    config: dict,
) -> dict:
    """
    Execute the graph with retry logic for transient failures.

    Args:
        initial_state: The initial agent state
        config: LangGraph config with thread_id

    Returns:
        Final state from graph execution

    Raises:
        TransientAgentError: For retriable errors (network, timeout)
        Exception: For non-retriable errors
    """
    try:
        return agent_graph.invoke(initial_state, config=config)
    except ConnectionError as e:
        logger.warning(f"[Agent] Transient connection error, will retry: {e}")
        raise TransientAgentError(str(e)) from e
    except TimeoutError as e:
        logger.warning(f"[Agent] Transient timeout error, will retry: {e}")
        raise TransientAgentError(str(e)) from e


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

    Uses LangGraph checkpointing for conversation persistence when session_id
    is provided. The checkpointer automatically stores and retrieves conversation
    state based on the thread_id.

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

    # Check query cache first (only for non-session queries to avoid stale context)
    cache_key = None
    if use_cache and not session_id:
        cache_key = _make_cache_key(question, mode, company_id)
        cached = _get_cached_result(cache_key)
        if cached:
            logger.info(f"[Agent] Cache hit, returning cached result")
            # Update latency to reflect cache hit
            cached_copy = cached.copy()
            cached_copy["meta"] = cached["meta"].copy()
            cached_copy["meta"]["latency_ms"] = int((time.time() - start_time) * 1000)
            cached_copy["meta"]["cached"] = True
            return cached_copy

    # Load conversation history from LangGraph checkpoint
    messages: list[Message] = []
    if session_id:
        checkpoint_state = get_session_state(session_id)
        if checkpoint_state:
            messages = checkpoint_state.get("messages", [])
            if messages:
                logger.debug(f"[Agent] Loaded {len(messages)} messages from checkpoint")

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

    # Build config with thread_id for LangGraph checkpointing
    # Always provide a thread_id (use session_id if provided, else generate one)
    thread_id = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    if session_id:
        logger.debug(f"[Agent] Using LangGraph checkpointing with thread_id={session_id}")

    # Run the graph with retry logic
    logger.info(f"[Agent] Starting graph execution for: {question[:50]}...")

    try:
        final_state = _execute_graph(initial_state, config)
    except TransientAgentError as e:
        # Retries exhausted
        logger.error(f"[Agent] Graph execution failed after retries: {e}")
        return _build_error_response(str(e), start_time)
    except Exception as e:
        logger.error(f"[Agent] Graph execution failed: {e}")
        return _build_error_response(str(e), start_time)

    latency_ms = int((time.time() - start_time) * 1000)

    # Audit logging
    audit = AgentAuditLogger()
    audit.log_query(
        question=question,
        mode_used=final_state.get("mode_used", "unknown"),
        company_id=final_state.get("resolved_company_id"),
        latency_ms=latency_ms,
        source_count=len(final_state.get("sources", [])),
        user_id=user_id,
        session_id=session_id,
    )

    logger.info(f"[Agent] Complete in {latency_ms}ms")

    # Build response with per-node latencies
    result = {
        "answer": final_state.get("answer", ""),
        "sources": [s.model_dump() if hasattr(s, 'model_dump') else s for s in final_state.get("sources", [])],
        "steps": final_state.get("steps", []),
        "raw_data": final_state.get("raw_data", {}),
        "follow_up_suggestions": final_state.get("follow_up_suggestions", []),
        "meta": {
            "mode_used": final_state.get("mode_used", "unknown"),
            "latency_ms": latency_ms,
            "company_id": final_state.get("resolved_company_id"),
            "intent": final_state.get("intent", "general"),
            "days": final_state.get("days", 90),
            # Per-node latency breakdown
            "router_latency_ms": final_state.get("router_latency_ms"),
            "fetch_latency_ms": final_state.get("fetch_latency_ms"),
            "answer_latency_ms": final_state.get("answer_latency_ms"),
            "followup_latency_ms": final_state.get("followup_latency_ms"),
        }
    }

    # Cache result for non-session queries
    if cache_key:
        _set_cached_result(cache_key, result)

    return result


def _build_error_response(error: str, start_time: float) -> dict:
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
        }
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
    """
    Backwards-compatible wrapper for run_agent.

    This maintains the same API as the previous orchestrator.
    """
    return run_agent(
        question=question,
        mode=mode,
        company_id=company_id,
        session_id=session_id,
        user_id=user_id,
    )


# =============================================================================
# Visualization
# =============================================================================

def get_graph_mermaid() -> str:
    """
    Get Mermaid diagram of the graph.

    Returns:
        Mermaid diagram string
    """
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


def print_graph_ascii() -> None:
    """Print ASCII representation of the graph."""
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │             LANGGRAPH AGENT (4 nodes)                       │
    └─────────────────────────────────────────────────────────────┘

                         ┌─────────┐
                         │  Route  │  (LLM structured output)
                         └────┬────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │       Fetch         │  (Parallel: CRM + Docs + Account)
                   └──────────┬──────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │   Answer    │  (LCEL chain)
                       └──────┬──────┘
                              │
                              ▼
                       ┌─────────────┐
                       │  Follow-up  │  (Structured output)
                       └──────┬──────┘
                              │
                              ▼
                          ┌───────┐
                          │  END  │
                          └───────┘
    """)


def get_checkpointer() -> MemorySaver:
    """Get the global checkpointer instance."""
    return _checkpointer


def get_session_state(session_id: str) -> dict | None:
    """
    Get the checkpointed state for a session.

    Args:
        session_id: The session/thread ID

    Returns:
        The stored state dict, or None if not found
    """
    try:
        config = {"configurable": {"thread_id": session_id}}
        checkpoint = _checkpointer.get(config)
        if checkpoint:
            return checkpoint.get("channel_values", {})
    except Exception as e:
        logger.warning(f"Failed to get session state: {e}")
    return None


__all__ = [
    "agent_graph",
    "run_agent",
    "answer_question",
    "build_agent_graph",
    "get_graph_mermaid",
    "print_graph_ascii",
    # Checkpointing
    "get_checkpointer",
    "get_session_state",
    # Cache
    "clear_query_cache",
    # Errors
    "TransientAgentError",
]


