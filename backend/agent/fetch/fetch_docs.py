"""Documentation RAG fetch node for LangGraph parallel execution."""

import logging
import time

from backend.agent.core.state import AgentState
from backend.agent.fetch.rag import call_docs_rag


logger = logging.getLogger(__name__)


def fetch_docs_node(state: AgentState) -> AgentState:
    """Fetch documentation via RAG."""
    start_time = time.time()
    question = state.get("question", "")

    logger.info("[FetchDocs] Searching documentation...")

    try:
        docs_answer, docs_sources = call_docs_rag(question)

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[FetchDocs] Complete in {latency_ms}ms, sources={len(docs_sources)}")

        return {
            "docs_answer": docs_answer,
            "sources": docs_sources,  # Uses reducer to merge with other sources
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[FetchDocs] Failed after {latency_ms}ms: {e}")
        return {"docs_answer": "", "error": f"Docs fetch failed: {e}"}


__all__ = ["fetch_docs_node"]
