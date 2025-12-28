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

import logging
import time
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.agent.state import AgentState
from backend.agent.nodes import (
    route_node,
    fetch_node,
    answer_node,
    followup_node,
)
from backend.agent.audit import AgentAuditLogger
from backend.agent.memory import get_conversation_history, add_message


logger = logging.getLogger(__name__)


# =============================================================================
# LangGraph Checkpointing
# =============================================================================

# Global checkpointer for conversation persistence
_checkpointer = MemorySaver()


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
# Runner Function
# =============================================================================

def run_agent(
    question: str,
    mode: str = "auto",
    company_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
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

    Returns:
        Dict matching ChatResponse schema
    """
    start_time = time.time()

    # Load conversation history for multi-turn support (legacy + fallback)
    messages = get_conversation_history(session_id)
    if messages:
        logger.debug(f"[Agent] Loaded {len(messages)} messages from session {session_id}")

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

    # Run the graph
    logger.info(f"[Agent] Starting graph execution for: {question[:50]}...")

    try:
        # Invoke with config for checkpointing
        final_state = agent_graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"[Agent] Graph execution failed: {e}")
        return {
            "answer": f"I'm sorry, I encountered an error: {str(e)}",
            "sources": [],
            "steps": [{"id": "error", "label": f"Error: {str(e)[:50]}", "status": "error"}],
            "raw_data": {},
            "follow_up_suggestions": [],
            "meta": {
                "mode_used": "error",
                "latency_ms": int((time.time() - start_time) * 1000),
            }
        }

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

    # Save conversation to memory for multi-turn support
    # TODO: This is redundant with LangGraph checkpointing. The dual storage exists because:
    #   - Legacy memory is used by get_last_company_context() for pronoun resolution
    #   - LangGraph MemorySaver checkpoints full state automatically
    # Future: Migrate fully to LangGraph checkpointing and remove legacy memory
    resolved_company = final_state.get("resolved_company_id")
    add_message(session_id, "user", question, resolved_company)
    add_message(session_id, "assistant", final_state.get("answer", ""), resolved_company)

    # Build response
    return {
        "answer": final_state.get("answer", ""),
        "sources": [s.model_dump() if hasattr(s, 'model_dump') else s for s in final_state.get("sources", [])],
        "steps": final_state.get("steps", []),
        "raw_data": final_state.get("raw_data", {}),
        "follow_up_suggestions": final_state.get("follow_up_suggestions", []),
        "meta": {
            "mode_used": final_state.get("mode_used", "unknown"),
            "latency_ms": latency_ms,
            "company_id": final_state.get("resolved_company_id"),
            "days": final_state.get("days", 90),
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
]


# =============================================================================
# CLI / Test
# =============================================================================

if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    print("\n" + "=" * 60)
    print("LangGraph Agent Workflow")
    print("=" * 60)

    print_graph_ascii()

    # Test if MOCK_LLM is set
    if os.environ.get("MOCK_LLM"):
        print("\n[MOCK MODE ENABLED]")

    print("\n" + "-" * 60)
    print("Testing Agent Graph")
    print("-" * 60)

    test_questions = [
        # Company-specific queries
        ("What's going on with Acme Manufacturing?", "auto"),

        # Documentation queries
        ("How do I create a new opportunity?", "auto"),

        # Renewals
        ("What renewals are coming up?", "auto"),

        # Pipeline summary (aggregate)
        ("What's the total pipeline value?", "auto"),

        # Contact search
        ("Who are the decision makers?", "auto"),

        # Company search
        ("Show me all enterprise accounts", "auto"),

        # Groups
        ("Who is in the at-risk accounts group?", "auto"),

        # Attachments
        ("Find all proposals", "auto"),
    ]

    for q, mode in test_questions:
        print(f"\nQ: {q}")
        print(f"Mode: {mode}")

        result = run_agent(q, mode=mode)

        print(f"  → Mode used: {result['meta']['mode_used']}")
        print(f"  → Company: {result['meta'].get('company_id', 'None')}")
        print(f"  → Sources: {len(result['sources'])}")
        print(f"  → Latency: {result['meta']['latency_ms']}ms")
        print(f"  → Steps: {[s['id'] for s in result['steps']]}")
        print(f"  → Answer: {result['answer'][:100]}...")
