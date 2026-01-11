"""Account context RAG fetch node for LangGraph sequential execution."""

import logging
import time

from backend.agent.core.state import AgentState
from backend.agent.fetch.rag import call_account_rag

logger = logging.getLogger(__name__)


def fetch_account_node(state: AgentState) -> AgentState:
    """
    Fetch account context via RAG when entity IDs are resolved.

    Entity IDs are resolved by fetch_crm from SQL query results.
    If any entity ID is resolved, RAG is called to fetch unstructured context.
    Filters by all resolved entities (company, contact, opportunity) for precise results.

    Skips RAG if needs_rag=False (set by slot planner for list/show questions).
    """
    start_time = time.time()

    # Check if RAG is needed (from slot planner decision)
    needs_rag = state.get("needs_rag", True)  # Default to True for backward compatibility
    if not needs_rag:
        logger.info("[FetchAccount] Skipped (needs_rag=False from planner)")
        return {"account_context_answer": "", "account_rag_invoked": False}

    # Build filters from all resolved entity IDs
    filters: dict[str, str] = {}
    company_id = state.get("resolved_company_id")
    contact_id = state.get("resolved_contact_id")
    opportunity_id = state.get("resolved_opportunity_id")

    if company_id:
        filters["company_id"] = company_id
    if contact_id:
        filters["contact_id"] = contact_id
    if opportunity_id:
        filters["opportunity_id"] = opportunity_id

    # Need at least one entity to filter by
    if not filters:
        logger.info("[FetchAccount] Skipped (no entity IDs resolved)")
        return {"account_context_answer": "", "account_rag_invoked": False}

    question = state.get("question", "")

    logger.info(f"[FetchAccount] Searching account context with filters={filters}...")

    try:
        account_answer, account_sources = call_account_rag(
            question=question,
            filters=filters,
        )

        # Split combined context back into individual chunks for RAGAS evaluation
        # The RAG tool joins chunks with "\n\n---\n\n"
        context_chunks = account_answer.split("\n\n---\n\n") if account_answer else []

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[FetchAccount] Complete in {latency_ms}ms, "
            f"sources={len(account_sources)}, chunks={len(context_chunks)}"
        )

        return {
            "account_context_answer": account_answer,
            "sources": account_sources,  # Uses reducer to merge with other sources
            "account_chunks": context_chunks,  # Individual chunks for RAGAS (account source)
            "account_rag_invoked": True,  # RAG was called (even if empty results)
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[FetchAccount] Failed after {latency_ms}ms: {e}")
        return {
            "account_context_answer": "",
            "account_rag_invoked": True,
            "error": f"Account fetch failed: {e}",
        }


__all__ = ["fetch_account_node"]
