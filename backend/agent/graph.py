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
from typing import TypedDict, Literal, Optional, Annotated
from operator import add

from langgraph.graph import StateGraph, END

from backend.agent.config import get_config
from backend.agent.schemas import Source, RouterResult
from backend.agent.llm_router import route_question
from backend.agent.audit import AgentAuditLogger
from backend.agent.progress import AgentProgress
from backend.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    COMPANY_NOT_FOUND_PROMPT,
    DATA_ANSWER_PROMPT,
)
from backend.agent.formatters import (
    format_company_section,
    format_activities_section,
    format_history_section,
    format_pipeline_section,
    format_renewals_section,
    format_docs_section,
)
from backend.agent.llm_helpers import (
    call_llm,
    call_docs_rag,
    generate_follow_up_suggestions,
)
from backend.agent.tools import (
    tool_company_lookup,
    tool_recent_activity,
    tool_recent_history,
    tool_pipeline,
    tool_upcoming_renewals,
)


logger = logging.getLogger(__name__)


# =============================================================================
# State Definition
# =============================================================================

class AgentState(TypedDict, total=False):
    """
    State that flows through the LangGraph workflow.
    
    This is the central data structure that each node can read/write.
    """
    # Input
    question: str
    mode: str  # "auto", "docs", "data", "data+docs"
    company_id: Optional[str]
    session_id: Optional[str]
    user_id: Optional[str]
    
    # Router output
    router_result: Optional[RouterResult]
    mode_used: str
    resolved_company_id: Optional[str]
    days: int
    intent: str
    
    # Data outputs
    company_data: Optional[dict]
    activities_data: Optional[dict]
    history_data: Optional[dict]
    pipeline_data: Optional[dict]
    renewals_data: Optional[dict]
    
    # Docs output
    docs_answer: str
    docs_sources: list[Source]
    
    # Sources accumulated from all steps (using reducer to append)
    sources: Annotated[list[Source], add]
    
    # Steps accumulated from all nodes (using reducer to append)
    steps: Annotated[list[dict], add]
    
    # Final outputs
    answer: str
    follow_up_suggestions: list[str]
    
    # Raw data for UI
    raw_data: dict
    
    # Error handling
    error: Optional[str]


# =============================================================================
# Node Functions
# =============================================================================

def route_node(state: AgentState) -> AgentState:
    """
    Router node: Determine mode and extract parameters.
    
    Uses LLM-based or heuristic routing based on config.
    """
    logger.info(f"[Route] Processing: {state['question'][:50]}...")
    
    try:
        router_result = route_question(
            state["question"],
            mode=state.get("mode", "auto"),
            company_id=state.get("company_id"),
        )
        
        logger.info(
            f"[Route] Result: mode={router_result.mode_used}, "
            f"company={router_result.company_id}, intent={router_result.intent}"
        )
        
        return {
            "router_result": router_result,
            "mode_used": router_result.mode_used,
            "resolved_company_id": router_result.company_id,
            "days": router_result.days,
            "intent": router_result.intent,
            "steps": [{"id": "router", "label": "Understanding your question", "status": "done"}],
        }
        
    except Exception as e:
        logger.error(f"[Route] Failed: {e}")
        return {
            "mode_used": "docs",  # Fallback to docs
            "days": 90,
            "intent": "general",
            "error": f"Routing failed: {e}",
            "steps": [{"id": "router", "label": "Understanding your question", "status": "error"}],
        }


def data_node(state: AgentState) -> AgentState:
    """
    Data node: Fetch CRM data based on router output.
    
    Fetches company info, activities, history, pipeline, and renewals.
    """
    logger.info(f"[Data] Fetching CRM data for intent={state.get('intent')}")
    
    sources: list[Source] = []
    raw_data = {
        "companies": [],
        "activities": [],
        "opportunities": [],
        "history": [],
        "renewals": [],
        "pipeline_summary": None,
    }
    
    company_data = None
    activities_data = None
    history_data = None
    pipeline_data = None
    renewals_data = None
    
    intent = state.get("intent", "general")
    resolved_company_id = state.get("resolved_company_id")
    days = state.get("days", 90)
    
    try:
        # Handle renewals intent (no specific company)
        if intent == "renewals" and not resolved_company_id:
            logger.debug(f"[Data] Fetching renewals for next {days} days")
            renewals_result = tool_upcoming_renewals(days=days)
            renewals_data = renewals_result.data
            sources.extend(renewals_result.sources)
            raw_data["renewals"] = renewals_data.get("renewals", [])[:8]
        
        # Handle company-specific queries
        elif resolved_company_id or state.get("router_result", {}).company_name_query if state.get("router_result") else None:
            query = resolved_company_id or (state.get("router_result").company_name_query if state.get("router_result") else None)
            logger.debug(f"[Data] Looking up company: {query}")
            
            company_result = tool_company_lookup(query or "")
            
            if company_result.data.get("found"):
                company_data = company_result.data
                sources.extend(company_result.sources)
                resolved_company_id = company_data["company"]["company_id"]
                raw_data["companies"] = [company_data["company"]]
                
                logger.debug(f"[Data] Fetching data for {resolved_company_id}")
                
                # Get activities
                activities_result = tool_recent_activity(resolved_company_id, days=days)
                activities_data = activities_result.data
                sources.extend(activities_result.sources)
                raw_data["activities"] = activities_data.get("activities", [])[:8]
                
                # Get history
                history_result = tool_recent_history(resolved_company_id, days=days)
                history_data = history_result.data
                sources.extend(history_result.sources)
                raw_data["history"] = history_data.get("history", [])[:8]
                
                # Get pipeline
                pipeline_result = tool_pipeline(resolved_company_id)
                pipeline_data = pipeline_result.data
                sources.extend(pipeline_result.sources)
                raw_data["opportunities"] = pipeline_data.get("opportunities", [])[:8]
                raw_data["pipeline_summary"] = pipeline_data.get("summary")
                
                logger.info(
                    f"[Data] Fetched: activities={len(activities_data.get('activities', []))}, "
                    f"history={len(history_data.get('history', []))}, "
                    f"opps={len(pipeline_data.get('opportunities', []))}"
                )
            else:
                company_data = company_result.data
                logger.info(f"[Data] Company not found: {query}")
        
        else:
            # No company specified - get general renewals
            logger.debug("[Data] No company specified, fetching general renewals")
            renewals_result = tool_upcoming_renewals(days=days)
            renewals_data = renewals_result.data
            sources.extend(renewals_result.sources)
            raw_data["renewals"] = renewals_data.get("renewals", [])[:8]
        
        return {
            "company_data": company_data,
            "activities_data": activities_data,
            "history_data": history_data,
            "pipeline_data": pipeline_data,
            "renewals_data": renewals_data,
            "resolved_company_id": resolved_company_id,
            "sources": sources,
            "raw_data": raw_data,
            "steps": [{"id": "data", "label": "Fetching CRM data", "status": "done"}],
        }
        
    except Exception as e:
        logger.error(f"[Data] Failed: {e}")
        return {
            "raw_data": raw_data,
            "steps": [{"id": "data", "label": "Fetching CRM data", "status": "error"}],
            "error": f"Data fetch failed: {e}",
        }


def docs_node(state: AgentState) -> AgentState:
    """
    Docs node: Fetch documentation via RAG.
    """
    logger.info(f"[Docs] Querying documentation...")
    
    try:
        docs_answer, docs_sources = call_docs_rag(state["question"])
        logger.info(f"[Docs] Retrieved {len(docs_sources)} sources")
        
        return {
            "docs_answer": docs_answer,
            "docs_sources": docs_sources,
            "sources": docs_sources,
            "steps": [{"id": "docs", "label": "Checking documentation", "status": "done"}],
        }
        
    except Exception as e:
        logger.error(f"[Docs] Failed: {e}")
        return {
            "docs_answer": "",
            "docs_sources": [],
            "steps": [{"id": "docs", "label": "Checking documentation", "status": "error"}],
        }


def skip_data_node(state: AgentState) -> AgentState:
    """Placeholder when data is skipped."""
    return {
        "raw_data": {
            "companies": [],
            "activities": [],
            "opportunities": [],
            "history": [],
            "renewals": [],
            "pipeline_summary": None,
        },
        "steps": [{"id": "data", "label": "Skipped (docs-only query)", "status": "skipped"}],
    }


def skip_docs_node(state: AgentState) -> AgentState:
    """Placeholder when docs is skipped."""
    return {
        "docs_answer": "",
        "docs_sources": [],
        "steps": [{"id": "docs", "label": "Skipped (data-only query)", "status": "skipped"}],
    }


def answer_node(state: AgentState) -> AgentState:
    """
    Answer node: Synthesize final answer using LLM.
    """
    logger.info("[Answer] Synthesizing response...")
    
    try:
        company_data = state.get("company_data")
        
        # Handle company not found case
        if company_data and not company_data.get("found"):
            matches_text = "\n".join([
                f"- {m.get('name')} ({m.get('company_id')})"
                for m in company_data.get("close_matches", [])[:5]
            ]) or "No similar companies found."
            
            prompt = COMPANY_NOT_FOUND_PROMPT.format(
                question=state["question"],
                query=company_data.get("query", "unknown"),
                matches=matches_text,
            )
            
            answer, llm_latency = call_llm(prompt, AGENT_SYSTEM_PROMPT)
        
        else:
            # Build context sections
            company_section = format_company_section(company_data)
            activities_section = format_activities_section(state.get("activities_data"))
            history_section = format_history_section(state.get("history_data"))
            pipeline_section = format_pipeline_section(state.get("pipeline_data"))
            renewals_section = format_renewals_section(state.get("renewals_data"))
            docs_section = format_docs_section(state.get("docs_answer", ""))
            
            prompt = DATA_ANSWER_PROMPT.format(
                question=state["question"],
                company_section=company_section,
                activities_section=activities_section,
                history_section=history_section,
                pipeline_section=pipeline_section,
                renewals_section=renewals_section,
                docs_section=docs_section,
            )
            
            answer, llm_latency = call_llm(prompt, AGENT_SYSTEM_PROMPT)
        
        logger.info(f"[Answer] Synthesized in {llm_latency}ms")
        
        return {
            "answer": answer,
            "steps": [{"id": "answer", "label": "Synthesizing answer", "status": "done"}],
        }
        
    except Exception as e:
        logger.error(f"[Answer] Failed: {e}")
        return {
            "answer": f"I encountered an error generating the answer: {str(e)}",
            "steps": [{"id": "answer", "label": "Synthesizing answer", "status": "error"}],
        }


def followup_node(state: AgentState) -> AgentState:
    """
    Follow-up node: Generate suggested follow-up questions.
    """
    config = get_config()
    
    if not config.enable_follow_up_suggestions:
        return {"follow_up_suggestions": []}
    
    logger.info("[Followup] Generating suggestions...")
    
    try:
        suggestions = generate_follow_up_suggestions(
            question=state["question"],
            mode=state.get("mode_used", "auto"),
            company_id=state.get("resolved_company_id"),
        )
        logger.debug(f"[Followup] Generated {len(suggestions)} suggestions")
        
        return {
            "follow_up_suggestions": suggestions,
            "steps": [{"id": "followup", "label": "Generating suggestions", "status": "done"}],
        }
        
    except Exception as e:
        logger.warning(f"[Followup] Failed: {e}")
        return {
            "follow_up_suggestions": [],
            "steps": [{"id": "followup", "label": "Generating suggestions", "status": "error"}],
        }


# =============================================================================
# Routing Logic
# =============================================================================

def route_by_mode(state: AgentState) -> Literal["data_only", "docs_only", "data_and_docs"]:
    """
    Conditional edge: Route based on mode_used.
    """
    mode = state.get("mode_used", "data+docs")
    
    if mode == "data":
        return "data_only"
    elif mode == "docs":
        return "docs_only"
    else:  # "data+docs" or fallback
        return "data_and_docs"


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
    company_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
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
    import time
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
    company_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
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


def print_graph_ascii():
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
        ("What's going on with Acme Manufacturing?", "auto"),
        ("How do I create a new opportunity?", "auto"),
        ("What renewals are coming up?", "auto"),
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
