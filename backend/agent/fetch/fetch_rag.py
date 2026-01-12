"""RAG fetch node for unstructured account context."""

import logging
import time

from backend.agent.core.state import AgentState, Source

logger = logging.getLogger(__name__)


def _call_account_rag(
    question: str,
    filters: dict[str, str],
) -> tuple[str, list[Source]]:
    """Call the account RAG tool with error handling.

    Args:
        question: The user's question
        filters: Dict of entity IDs to filter by (company_id, contact_id, opportunity_id)

    Returns:
        Tuple of (context string, list of sources)
    """
    try:
        from backend.agent.rag.tools import tool_entity_rag

        return tool_entity_rag(question, filters)
    except Exception as e:
        logger.warning(f"Account RAG failed: {e}")
        return "", []


def fetch_rag_node(state: AgentState) -> AgentState:
    """Fetch unstructured context via RAG using resolved entity IDs from fetch_sql."""
    start_time = time.time()

    # Check if RAG is needed (from slot planner decision)
    needs_rag = state.get("needs_rag", True)  # Default to True for backward compatibility
    if not needs_rag:
        logger.info("[FetchRAG] Skipped (needs_rag=False from planner)")
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
        logger.info("[FetchRAG] Skipped (no entity IDs resolved)")
        return {"account_context_answer": "", "account_rag_invoked": False}

    question = state.get("question", "")

    logger.info(f"[FetchRAG] Searching account context with filters={filters}...")

    try:
        account_answer, account_sources = _call_account_rag(
            question=question,
            filters=filters,
        )

        # Split combined context back into individual chunks for RAGAS evaluation
        # The RAG tool joins chunks with "\n\n---\n\n"
        context_chunks = account_answer.split("\n\n---\n\n") if account_answer else []

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[FetchRAG] Complete in {latency_ms}ms, "
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
        logger.error(f"[FetchRAG] Failed after {latency_ms}ms: {e}")
        return {
            "account_context_answer": "",
            "account_rag_invoked": True,
            "error": f"Account fetch failed: {e}",
        }


__all__ = ["fetch_rag_node"]
