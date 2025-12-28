"""
LangGraph node functions for agent workflow.

Simplified 4-node architecture:
  Route → Fetch (parallel) → Answer → Followup

The Fetch node always runs CRM data + Docs RAG in parallel,
with optional Account RAG for company-specific intents.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from backend.agent.state import AgentState
from backend.agent.config import get_config
from backend.agent.llm_router import route_question
from backend.agent.memory import format_history_for_prompt
from backend.agent.formatters import (
    format_company_section,
    format_activities_section,
    format_history_section,
    format_pipeline_section,
    format_renewals_section,
    format_docs_section,
    format_account_context_section,
    format_conversation_history_section,
)
from backend.agent.llm_helpers import (
    call_docs_rag,
    call_account_rag,
    generate_follow_up_suggestions,
    call_answer_chain,
    call_not_found_chain,
)
from backend.agent.intent_handlers import IntentContext, dispatch_intent


logger = logging.getLogger(__name__)


# =============================================================================
# Router Node
# =============================================================================

def route_node(state: AgentState) -> AgentState:
    """
    Router node: Determine mode and extract parameters.

    Uses LLM-based or heuristic routing based on config.
    Passes conversation history for pronoun resolution.
    """
    logger.info(f"[Route] Processing: {state['question'][:50]}...")

    # Format conversation history for the router
    messages = state.get("messages", [])
    conversation_history = format_history_for_prompt(messages) if messages else ""

    try:
        router_result = route_question(
            state["question"],
            mode=state.get("mode", "auto"),
            company_id=state.get("company_id"),
            conversation_history=conversation_history,
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


# =============================================================================
# Fetch Node (Unified Parallel Fetch)
# =============================================================================

def fetch_node(state: AgentState) -> AgentState:
    """
    Unified fetch node: Fetch CRM data, docs, and account context in parallel.

    Always fetches both CRM data and documentation concurrently using
    ThreadPoolExecutor. For company-specific intents, also fetches
    Account RAG (notes, attachments) in parallel.

    This unified approach simplifies the graph while maintaining
    the same latency characteristics (LLM synthesis dominates).
    """
    logger.info("[Fetch] Fetching data, docs, and account context in parallel...")

    # Prepare results containers
    data_result = {}
    docs_result = {}
    account_result = {}
    errors = []

    # Check if we should fetch account context (notes, attachments via Account RAG)
    # Trigger for company-specific intents where unstructured text might be relevant
    intent = state.get("intent", "general")
    company_id = state.get("resolved_company_id")
    should_fetch_account_context = (
        company_id and
        intent in ("account_context", "company_status", "history", "pipeline")
    )

    def fetch_data():
        """Fetch CRM data (runs in thread)."""
        try:
            ctx = IntentContext(
                question=state.get("question", "").lower(),
                resolved_company_id=state.get("resolved_company_id"),
                days=state.get("days", 90),
                router_result=state.get("router_result"),
            )
            result = dispatch_intent(intent, ctx)
            return {
                "company_data": result.company_data,
                "activities_data": result.activities_data,
                "history_data": result.history_data,
                "pipeline_data": result.pipeline_data,
                "renewals_data": result.renewals_data,
                "contacts_data": result.contacts_data,
                "groups_data": result.groups_data,
                "attachments_data": result.attachments_data,
                "resolved_company_id": result.resolved_company_id or ctx.resolved_company_id,
                "sources": result.sources,
                "raw_data": result.raw_data,
            }
        except Exception as e:
            logger.error(f"[Fetch] Data fetch failed: {e}")
            return {"error": str(e)}

    def fetch_docs():
        """Fetch docs via RAG (runs in thread)."""
        try:
            docs_answer, docs_sources = call_docs_rag(state["question"])
            return {
                "docs_answer": docs_answer,
                "docs_sources": docs_sources,
            }
        except Exception as e:
            logger.error(f"[Fetch] Docs fetch failed: {e}")
            return {"docs_answer": "", "docs_sources": [], "error": str(e)}

    def fetch_account_context():
        """Fetch account context via Account RAG (runs in thread)."""
        try:
            if not company_id:
                return {"account_context_answer": "", "account_context_sources": []}

            account_answer, account_sources = call_account_rag(
                question=state["question"],
                company_id=company_id,
            )
            return {
                "account_context_answer": account_answer,
                "account_context_sources": account_sources,
            }
        except Exception as e:
            logger.error(f"[Fetch] Account RAG failed: {e}")
            return {"account_context_answer": "", "account_context_sources": [], "error": str(e)}

    # Execute in parallel using ThreadPoolExecutor
    max_workers = 3 if should_fetch_account_context else 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        futures["data"] = executor.submit(fetch_data)
        futures["docs"] = executor.submit(fetch_docs)

        if should_fetch_account_context:
            futures["account"] = executor.submit(fetch_account_context)

        # Collect results
        for name, future in futures.items():
            try:
                result = future.result()
                if name == "data":
                    data_result = result
                elif name == "docs":
                    docs_result = result
                elif name == "account":
                    account_result = result
            except Exception as e:
                errors.append(f"{name}: {str(e)}")

    # Merge results
    all_sources = (
        data_result.get("sources", []) +
        docs_result.get("docs_sources", []) +
        account_result.get("account_context_sources", [])
    )

    steps = [
        {"id": "data", "label": "Fetching CRM data", "status": "done" if "error" not in data_result else "error"},
        {"id": "docs", "label": "Checking documentation", "status": "done" if "error" not in docs_result else "error"},
    ]

    if should_fetch_account_context:
        steps.append({
            "id": "account_context",
            "label": "Searching account notes",
            "status": "done" if "error" not in account_result else "error"
        })

    merged = {
        # Data results
        "company_data": data_result.get("company_data"),
        "activities_data": data_result.get("activities_data"),
        "history_data": data_result.get("history_data"),
        "pipeline_data": data_result.get("pipeline_data"),
        "renewals_data": data_result.get("renewals_data"),
        "contacts_data": data_result.get("contacts_data"),
        "groups_data": data_result.get("groups_data"),
        "attachments_data": data_result.get("attachments_data"),
        "resolved_company_id": data_result.get("resolved_company_id"),
        "raw_data": data_result.get("raw_data", {}),
        # Docs results
        "docs_answer": docs_result.get("docs_answer", ""),
        "docs_sources": docs_result.get("docs_sources", []),
        # Account RAG results
        "account_context_answer": account_result.get("account_context_answer", ""),
        "account_context_sources": account_result.get("account_context_sources", []),
        # Combined sources
        "sources": all_sources,
        # Steps
        "steps": steps,
    }

    if errors:
        merged["error"] = "; ".join(errors)

    logger.info(
        f"[Fetch] Parallel fetch complete: {len(all_sources)} sources "
        f"(account_context={'yes' if should_fetch_account_context else 'no'})"
    )
    return merged


# =============================================================================
# Answer Node
# =============================================================================

def answer_node(state: AgentState) -> AgentState:
    """
    Answer node: Synthesize final answer using LCEL chains.

    Uses LangChain LCEL chains (prompt | llm | parser) for answer synthesis.
    """
    logger.info("[Answer] Synthesizing response with LCEL chain...")

    try:
        company_data = state.get("company_data")

        # Handle company not found case
        if company_data and not company_data.get("found"):
            matches_text = "\n".join([
                f"- {m.get('name')} ({m.get('company_id')})"
                for m in company_data.get("close_matches", [])[:5]
            ]) or "No similar companies found."

            # Use LCEL chain for company not found
            answer, llm_latency = call_not_found_chain(
                question=state["question"],
                query=company_data.get("query", "unknown"),
                matches=matches_text,
            )

        else:
            # Build context sections
            conversation_history_section = format_conversation_history_section(
                state.get("messages", [])
            )
            company_section = format_company_section(company_data)
            activities_section = format_activities_section(state.get("activities_data"))
            history_section = format_history_section(state.get("history_data"))
            pipeline_section = format_pipeline_section(state.get("pipeline_data"))
            renewals_section = format_renewals_section(state.get("renewals_data"))
            docs_section = format_docs_section(state.get("docs_answer", ""))
            account_context_section = format_account_context_section(
                state.get("account_context_answer", "")
            )

            # Use LCEL chain for answer synthesis
            answer, llm_latency = call_answer_chain(
                question=state["question"],
                conversation_history_section=conversation_history_section,
                company_section=company_section,
                activities_section=activities_section,
                history_section=history_section,
                pipeline_section=pipeline_section,
                renewals_section=renewals_section,
                docs_section=docs_section,
                account_context_section=account_context_section,
            )

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


# =============================================================================
# Follow-up Node
# =============================================================================

def followup_node(state: AgentState) -> AgentState:
    """
    Follow-up node: Generate suggested follow-up questions.

    Uses conversation history for contextual suggestions.
    """
    config = get_config()

    if not config.enable_follow_up_suggestions:
        return {"follow_up_suggestions": []}

    logger.info("[Followup] Generating suggestions...")

    # Format conversation history for follow-up context
    messages = state.get("messages", [])
    conversation_history = format_history_for_prompt(messages) if messages else ""

    try:
        suggestions = generate_follow_up_suggestions(
            question=state["question"],
            mode=state.get("mode_used", "auto"),
            company_id=state.get("resolved_company_id"),
            conversation_history=conversation_history,
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


__all__ = [
    "route_node",
    "fetch_node",
    "answer_node",
    "followup_node",
]
