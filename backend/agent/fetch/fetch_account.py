"""Account context RAG fetch node for LangGraph parallel execution."""

import logging
import time

from backend.agent.core.state import AgentState
from backend.agent.fetch.rag import call_account_rag

logger = logging.getLogger(__name__)


# Intents that trigger Account RAG for unstructured text search
ACCOUNT_RAG_INTENTS = frozenset({
    "account_context",
    "company_status",
    "history",
    "pipeline",
})


def fetch_account_node(state: AgentState) -> AgentState:
    """Fetch account context via RAG (conditional on intent and company_id)."""
    start_time = time.time()

    intent = state.get("intent", "general")
    company_id = state.get("resolved_company_id")
    question = state.get("question", "")

    # Skip if no company_id or wrong intent
    if not company_id or intent not in ACCOUNT_RAG_INTENTS:
        logger.info(f"[FetchAccount] Skipped (company={company_id}, intent={intent})")
        return {"account_context_answer": ""}

    logger.info(f"[FetchAccount] Searching account context for {company_id}...")

    try:
        account_answer, account_sources = call_account_rag(
            question=question,
            company_id=company_id,
        )

        # Split combined context back into individual chunks for RAGAS evaluation
        # The RAG tool joins chunks with "\n\n---\n\n"
        context_chunks = account_answer.split("\n\n---\n\n") if account_answer else []

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[FetchAccount] Complete in {latency_ms}ms, sources={len(account_sources)}, chunks={len(context_chunks)}")

        return {
            "account_context_answer": account_answer,
            "sources": account_sources,  # Uses reducer to merge with other sources
            "context_chunks": context_chunks,  # Individual chunks for RAGAS
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[FetchAccount] Failed after {latency_ms}ms: {e}")
        return {"account_context_answer": "", "error": f"Account fetch failed: {e}"}


__all__ = ["fetch_account_node"]
