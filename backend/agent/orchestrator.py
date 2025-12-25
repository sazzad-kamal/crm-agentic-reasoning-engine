"""
Agent orchestration for answering CRM questions.

This module now uses LangGraph for workflow management.
See backend/agent/graph.py for the graph implementation.

For backwards compatibility, this module re-exports:
- answer_question: Main entry point
- run_agent: Graph-based runner
- agent_graph: Compiled LangGraph instance
"""

# Re-export from graph module for backwards compatibility
from backend.agent.graph import (
    answer_question,
    run_agent,
    agent_graph,
    print_graph_ascii,
    get_graph_mermaid,
    AgentState,
)

__all__ = [
    "answer_question",
    "run_agent", 
    "agent_graph",
    "print_graph_ascii",
    "get_graph_mermaid",
    "AgentState",
]


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import os
    import logging
    
    # Enable logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    
    # Enable mock mode for testing
    os.environ["MOCK_LLM"] = "1"
    
    print("Testing Agent (LangGraph)")
    print("=" * 60)
    
    print_graph_ascii()
    
    questions = [
        "What's going on with Acme Manufacturing in the last 90 days?",
        "Which accounts have upcoming renewals in the next 90 days?",
        "How do I create a new opportunity?",
    ]
    
    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print("-" * 60)
        
        result = answer_question(q)
        
        print(f"Mode: {result['meta']['mode_used']}")
        print(f"Company: {result['meta'].get('company_id')}")
        print(f"Latency: {result['meta']['latency_ms']}ms")
        print(f"\nSteps:")
        for step in result['steps']:
            print(f"  - {step['id']}: {step['label']} [{step['status']}]")
        print(f"\nSources ({len(result['sources'])}):")
        for src in result['sources'][:3]:
            print(f"  - {src['type']}: {src['label']}")
        print(f"\nAnswer:\n{result['answer'][:300]}...")
