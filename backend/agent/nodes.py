"""
LangGraph node functions for agent workflow.

Simplified 4-node architecture:
  Route → Fetch (parallel) → Answer → Followup

The Fetch node always runs CRM data + Docs RAG in parallel,
with optional Account RAG for company-specific intents.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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
    format_contacts_section,
    format_groups_section,
    format_attachments_section,
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
# Constants
# =============================================================================

# Intents that trigger Account RAG for unstructured text search
ACCOUNT_RAG_INTENTS = frozenset({
    "account_context",
    "company_status",
    "history",
    "pipeline",
})


# =============================================================================
# Fetch Helper Functions (module-level for testability)
# =============================================================================

def _fetch_crm_data(
    question: str,
    intent: str,
    resolved_company_id: str | None,
    days: int,
    router_result: object | None,
) -> dict:
    """
    Fetch CRM data based on intent.

    Extracted to module level for testability.
    """
    try:
        ctx = IntentContext(
            question=question.lower(),
            resolved_company_id=resolved_company_id,
            days=days,
            router_result=router_result,
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


def _fetch_docs(question: str) -> dict:
    """
    Fetch documentation via RAG.

    Extracted to module level for testability.
    """
    try:
        docs_answer, docs_sources = call_docs_rag(question)
        return {
            "docs_answer": docs_answer,
            "docs_sources": docs_sources,
        }
    except Exception as e:
        logger.error(f"[Fetch] Docs fetch failed: {e}")
        return {"docs_answer": "", "docs_sources": [], "error": str(e)}


def _fetch_account_context(question: str, company_id: str) -> dict:
    """
    Fetch account context via Account RAG.

    Extracted to module level for testability.
    """
    try:
        account_answer, account_sources = call_account_rag(
            question=question,
            company_id=company_id,
        )
        return {
            "account_context_answer": account_answer,
            "account_context_sources": account_sources,
        }
    except Exception as e:
        logger.error(f"[Fetch] Account RAG failed: {e}")
        return {"account_context_answer": "", "account_context_sources": [], "error": str(e)}


# =============================================================================
# Router Node
# =============================================================================

def route_node(state: AgentState) -> AgentState:
    """
    Router node: Determine mode and extract parameters.

    Uses LLM-based or heuristic routing based on config.
    Passes conversation history for pronoun resolution.
    """
    config = get_config()
    start_time = time.time()

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

        # Validate router_result has required fields
        if not hasattr(router_result, "mode_used") or not router_result.mode_used:
            raise ValueError("Router returned invalid result: missing mode_used")

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[Route] Result: mode={router_result.mode_used}, "
            f"company={router_result.company_id}, intent={router_result.intent}, "
            f"latency={latency_ms}ms"
        )

        return {
            "router_result": router_result,
            "mode_used": router_result.mode_used,
            "resolved_company_id": router_result.company_id,
            "days": router_result.days or config.default_days,
            "intent": router_result.intent or "general",
            "router_latency_ms": latency_ms,
            "steps": [{
                "id": "router",
                "label": "Understanding your question",
                "status": "done",
                "latency_ms": latency_ms,
            }],
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[Route] Failed after {latency_ms}ms: {e}")

        # Determine fallback mode based on question content
        question_lower = state["question"].lower()
        fallback_mode = "docs"  # Default fallback
        if any(kw in question_lower for kw in ["company", "account", "customer", "pipeline", "renewal"]):
            fallback_mode = "data"

        return {
            "mode_used": fallback_mode,
            "days": config.default_days,
            "intent": "general",
            "router_latency_ms": latency_ms,
            "error": f"Routing failed: {e}",
            "steps": [{
                "id": "router",
                "label": "Understanding your question",
                "status": "error",
                "latency_ms": latency_ms,
            }],
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
    config = get_config()
    start_time = time.time()

    logger.info("[Fetch] Fetching data, docs, and account context in parallel...")

    # Prepare results containers
    data_result = {}
    docs_result = {}
    account_result = {}
    errors = []

    # Check if we should fetch account context (notes, attachments via Account RAG)
    intent = state.get("intent", "general")
    company_id = state.get("resolved_company_id")
    should_fetch_account_context = (
        company_id is not None and
        intent in ACCOUNT_RAG_INTENTS
    )

    # Get state values for fetch functions
    question = state.get("question", "")
    days = state.get("days", config.default_days)
    router_result = state.get("router_result")
    timeout = config.fetch_timeout_seconds

    # Execute in parallel using ThreadPoolExecutor
    max_workers = 3 if should_fetch_account_context else 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        futures["data"] = executor.submit(
            _fetch_crm_data,
            question=question,
            intent=intent,
            resolved_company_id=company_id,
            days=days,
            router_result=router_result,
        )
        futures["docs"] = executor.submit(_fetch_docs, question=question)

        if should_fetch_account_context:
            futures["account"] = executor.submit(
                _fetch_account_context,
                question=question,
                company_id=company_id,
            )

        # Collect results with timeout
        for name, future in futures.items():
            try:
                result = future.result(timeout=timeout)
                if name == "data":
                    data_result = result
                elif name == "docs":
                    docs_result = result
                elif name == "account":
                    account_result = result
            except FuturesTimeoutError:
                logger.error(f"[Fetch] {name} timed out after {timeout}s")
                errors.append(f"{name}: timeout after {timeout}s")
            except Exception as e:
                logger.error(f"[Fetch] {name} failed: {e}")
                errors.append(f"{name}: {str(e)}")

    # Merge results
    all_sources = (
        data_result.get("sources", []) +
        docs_result.get("docs_sources", []) +
        account_result.get("account_context_sources", [])
    )

    latency_ms = int((time.time() - start_time) * 1000)

    steps = [
        {
            "id": "data",
            "label": "Fetching CRM data",
            "status": "done" if "error" not in data_result else "error",
        },
        {
            "id": "docs",
            "label": "Checking documentation",
            "status": "done" if "error" not in docs_result else "error",
        },
    ]

    if should_fetch_account_context:
        steps.append({
            "id": "account_context",
            "label": "Searching account notes",
            "status": "done" if "error" not in account_result else "error",
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
        # Latency
        "fetch_latency_ms": latency_ms,
        # Steps
        "steps": steps,
    }

    if errors:
        merged["error"] = "; ".join(errors)

    logger.info(
        f"[Fetch] Parallel fetch complete in {latency_ms}ms: {len(all_sources)} sources "
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
    config = get_config()
    start_time = time.time()

    logger.info("[Answer] Synthesizing response with LCEL chain...")

    try:
        company_data = state.get("company_data")
        llm_latency = 0

        # Handle company not found case
        if company_data and not company_data.get("found"):
            close_matches = company_data.get("close_matches", [])[:config.max_close_matches]
            matches_text = "\n".join([
                f"- {m.get('name')} ({m.get('company_id')})"
                for m in close_matches
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
            contacts_section = format_contacts_section(state.get("contacts_data"))
            activities_section = format_activities_section(state.get("activities_data"))
            history_section = format_history_section(state.get("history_data"))
            pipeline_section = format_pipeline_section(state.get("pipeline_data"))
            renewals_section = format_renewals_section(state.get("renewals_data"))
            groups_section = format_groups_section(state.get("groups_data"))
            attachments_section = format_attachments_section(state.get("attachments_data"))
            docs_section = format_docs_section(state.get("docs_answer", ""))
            account_context_section = format_account_context_section(
                state.get("account_context_answer", "")
            )

            # Use LCEL chain for answer synthesis
            answer, llm_latency = call_answer_chain(
                question=state["question"],
                conversation_history_section=conversation_history_section,
                company_section=company_section,
                contacts_section=contacts_section,
                activities_section=activities_section,
                history_section=history_section,
                pipeline_section=pipeline_section,
                renewals_section=renewals_section,
                groups_section=groups_section,
                attachments_section=attachments_section,
                docs_section=docs_section,
                account_context_section=account_context_section,
            )

        # Validate answer
        if not answer or not answer.strip():
            logger.warning("[Answer] LLM returned empty answer, using fallback")
            answer = "I apologize, but I wasn't able to generate a complete response. Please try rephrasing your question."

        total_latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[Answer] Synthesized in {total_latency_ms}ms (LLM: {llm_latency}ms)")

        # Update messages for conversation memory (persisted via LangGraph checkpoint)
        messages = list(state.get("messages", []))
        messages.append({
            "role": "user",
            "content": state["question"],
            "company_id": state.get("resolved_company_id"),
        })
        messages.append({
            "role": "assistant",
            "content": answer,
            "company_id": state.get("resolved_company_id"),
        })

        return {
            "answer": answer,
            "messages": messages,  # Updated messages for next turn
            "answer_latency_ms": total_latency_ms,
            "llm_latency_ms": llm_latency,
            "steps": [{
                "id": "answer",
                "label": "Synthesizing answer",
                "status": "done",
                "latency_ms": total_latency_ms,
            }],
        }

    except Exception as e:
        total_latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[Answer] Failed after {total_latency_ms}ms: {e}")
        error_answer = f"I encountered an error generating the answer: {str(e)}"

        # Still update messages for conversation continuity
        messages = list(state.get("messages", []))
        messages.append({
            "role": "user",
            "content": state["question"],
            "company_id": state.get("resolved_company_id"),
        })
        messages.append({
            "role": "assistant",
            "content": error_answer,
            "company_id": state.get("resolved_company_id"),
        })

        return {
            "answer": error_answer,
            "messages": messages,
            "answer_latency_ms": total_latency_ms,
            "llm_latency_ms": 0,
            "error": str(e),
            "steps": [{
                "id": "answer",
                "label": "Synthesizing answer",
                "status": "error",
                "latency_ms": total_latency_ms,
            }],
        }


# =============================================================================
# Follow-up Node
# =============================================================================

def followup_node(state: AgentState) -> AgentState:
    """
    Follow-up node: Generate suggested follow-up questions.

    Uses conversation history and available data for grounded suggestions.
    """
    config = get_config()
    start_time = time.time()

    if not config.enable_follow_up_suggestions:
        logger.info("[Followup] Suggestions disabled in config")
        return {"follow_up_suggestions": []}

    logger.info("[Followup] Generating suggestions...")

    # Format conversation history for follow-up context
    messages = state.get("messages", [])
    conversation_history = format_history_for_prompt(messages) if messages else ""

    # Extract company name from company_data
    company_data = state.get("company_data", {})
    company_name = None
    if company_data and company_data.get("found"):
        company_info = company_data.get("company", {})
        company_name = company_info.get("name")

    # Build available data counts from raw_data
    raw_data = state.get("raw_data", {})
    available_data = {
        "contacts": len(raw_data.get("contacts", [])) if isinstance(raw_data, dict) else 0,
        "activities": len(raw_data.get("activities", [])) if isinstance(raw_data, dict) else 0,
        "opportunities": len(raw_data.get("opportunities", [])) if isinstance(raw_data, dict) else 0,
        "history": len(raw_data.get("history", [])) if isinstance(raw_data, dict) else 0,
        "renewals": len(raw_data.get("renewals", [])) if isinstance(raw_data, dict) else 0,
        "pipeline_summary": raw_data.get("pipeline_summary") if isinstance(raw_data, dict) else None,
        "docs": len(state.get("doc_sources", [])),
    }

    try:
        suggestions = generate_follow_up_suggestions(
            question=state["question"],
            mode=state.get("mode_used", "auto"),
            company_id=state.get("resolved_company_id"),
            company_name=company_name,
            conversation_history=conversation_history,
            available_data=available_data,
        )

        # Validate and limit suggestions
        if suggestions:
            suggestions = [s for s in suggestions if s and s.strip()]
            suggestions = suggestions[:config.max_followup_suggestions]

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[Followup] Generated {len(suggestions)} suggestions in {latency_ms}ms")

        return {
            "follow_up_suggestions": suggestions,
            "followup_latency_ms": latency_ms,
            "steps": [{
                "id": "followup",
                "label": "Generating suggestions",
                "status": "done",
                "latency_ms": latency_ms,
            }],
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"[Followup] Failed after {latency_ms}ms: {e}")
        return {
            "follow_up_suggestions": [],
            "followup_latency_ms": latency_ms,
            "steps": [{
                "id": "followup",
                "label": "Generating suggestions",
                "status": "error",
                "latency_ms": latency_ms,
            }],
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Node functions
    "route_node",
    "fetch_node",
    "answer_node",
    "followup_node",
    # Constants (for testing)
    "ACCOUNT_RAG_INTENTS",
    # Helper functions (for testing)
    "_fetch_crm_data",
    "_fetch_docs",
    "_fetch_account_context",
]
