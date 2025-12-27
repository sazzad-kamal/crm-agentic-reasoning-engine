"""
LangGraph-based agent orchestration.

Implements a graph workflow for answering CRM questions:

    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │  Route  │  Determine mode & extract entities
    └────┬────┘
         │
         ├─────────────────┬─────────────────┐
         ▼                 ▼                 ▼
    ┌─────────┐      ┌─────────┐      ┌─────────────┐
    │  Data   │      │  Docs   │      │ Data + Docs │
    │  Only   │      │  Only   │      │   (Both)    │
    └────┬────┘      └────┬────┘      └──────┬──────┘
         │                 │                 │
         └─────────────────┴─────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Answer    │  Synthesize with LLM
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Follow-up  │  Generate suggestions
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

from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.nodes import (
    route_node,
    data_node,
    docs_node,
    skip_data_node,
    skip_docs_node,
    answer_node,
    followup_node,
    route_by_mode,
)
from backend.agent.audit import AgentAuditLogger


logger = logging.getLogger(__name__)


# =============================================================================
# Graph Construction
# =============================================================================

def build_agent_graph() -> StateGraph:
    """
    Build the LangGraph workflow.

    Returns compiled graph ready for execution.
    """
    # Create graph with state schema
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route", route_node)
    graph.add_node("data", data_node)
    graph.add_node("docs", docs_node)
    graph.add_node("skip_data", skip_data_node)
    graph.add_node("skip_docs", skip_docs_node)
    graph.add_node("answer", answer_node)
    graph.add_node("followup", followup_node)

    # Set entry point
    graph.set_entry_point("route")

    # Add conditional routing after route node
    # Routes to: data (for data or data+docs), skip_data (for docs-only)
    graph.add_conditional_edges(
        "route",
        route_by_mode,
        {
            "data_only": "data",
            "docs_only": "skip_data",
            "data_and_docs": "data",
        }
    )

    # After skip_data, go to docs
    graph.add_edge("skip_data", "docs")

    # Data path - conditionally go to docs or skip_docs
    graph.add_conditional_edges(
        "data",
        lambda s: "docs" if s.get("mode_used") == "data+docs" else "skip_docs",
        {
            "docs": "docs",
            "skip_docs": "skip_docs",
        }
    )

    # Docs leads to answer
    graph.add_edge("docs", "answer")

    # Skip docs leads to answer
    graph.add_edge("skip_docs", "answer")

    # Answer leads to followup
    graph.add_edge("answer", "followup")

    # Followup is the end
    graph.add_edge("followup", END)

    return graph.compile()


# Compile the graph once at module load
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

    Args:
        question: The user's question
        mode: Mode override ("auto", "docs", "data", "data+docs")
        company_id: Pre-specified company ID
        session_id: Optional session ID
        user_id: Optional user ID

    Returns:
        Dict matching ChatResponse schema
    """
    start_time = time.time()

    # Initialize state
    initial_state: AgentState = {
        "question": question,
        "mode": mode,
        "company_id": company_id,
        "session_id": session_id,
        "user_id": user_id,
        "sources": [],
        "steps": [],
        "raw_data": {},
        "follow_up_suggestions": [],
    }

    # Run the graph
    logger.info(f"[Agent] Starting graph execution for: {question[:50]}...")

    try:
        final_state = agent_graph.invoke(initial_state)
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
    route -->|data| data[Fetch CRM Data]
    route -->|docs| skip_data[Skip Data]
    route -->|data+docs| data

    data -->|data+docs| docs[Fetch Docs]
    data -->|data only| skip_docs[Skip Docs]

    skip_data --> docs
    docs --> answer[Synthesize Answer]
    skip_docs --> answer

    answer --> followup[Generate Follow-ups]
    followup --> END((End))

    style route fill:#e1f5fe
    style data fill:#fff3e0
    style docs fill:#f3e5f5
    style answer fill:#e8f5e9
    style followup fill:#fce4ec
"""


def print_graph_ascii() -> None:
    """Print ASCII representation of the graph."""
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                      LANGGRAPH AGENT                        │
    └─────────────────────────────────────────────────────────────┘

                           ┌─────────┐
                           │  START  │
                           └────┬────┘
                                │
                                ▼
                           ┌─────────┐
                           │  Route  │
                           └────┬────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
       ┌─────────┐        ┌─────────┐        ┌───────────┐
       │  Data   │        │  Docs   │        │Data + Docs│
       │  Only   │        │  Only   │        │  (Both)   │
       └────┬────┘        └────┬────┘        └─────┬─────┘
            │                  │                   │
            │                  │           ┌───────┴───────┐
            │                  │           ▼               ▼
            │                  │      ┌─────────┐    ┌─────────┐
            │                  │      │  Data   │───▶│  Docs   │
            │                  │      └─────────┘    └────┬────┘
            │                  │                          │
            └──────────────────┴──────────────────────────┘
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
    """)


__all__ = [
    "agent_graph",
    "run_agent",
    "answer_question",
    "build_agent_graph",
    "get_graph_mermaid",
    "print_graph_ascii",
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
