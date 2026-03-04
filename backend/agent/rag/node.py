"""RAG node for LangGraph workflow.

Handles documentation-related queries by searching Act! CRM docs
and synthesizing grounded answers.
"""

import logging

from backend.agent.rag.retriever import retrieve_and_answer
from backend.agent.state import AgentState

logger = logging.getLogger(__name__)


def rag_node(state: AgentState) -> dict:
    """RAG node: retrieve documentation and synthesize answer.

    This node is invoked when the user asks "how to" questions
    about Act! CRM functionality.

    Args:
        state: Current agent state with question

    Returns:
        Updated state with RAG results in sql_results
    """
    question = state.get("question", "")
    logger.info(f"[RAG] Processing: {question[:50]}...")

    try:
        result = retrieve_and_answer(question, top_k=5)

        # Format sources for evidence display
        evidence = []
        for i, source in enumerate(result.sources[:3], 1):
            evidence.append({
                "id": f"D{i}",
                "source": source["source"],
                "excerpt": source["text"][:200] + "..." if len(source["text"]) > 200 else source["text"],
                "score": source["score"],
            })

        logger.info(
            f"[RAG] Retrieved {len(result.sources)} sources, "
            f"confidence={result.confidence:.2f}"
        )

        return {
            "sql_results": {
                "rag_answer": result.answer,
                "rag_sources": evidence,
                "rag_confidence": result.confidence,
                "_source": "documentation",
            }
        }

    except Exception as e:
        logger.error(f"[RAG] Failed: {e}")
        return {
            "sql_results": {
                "rag_answer": None,
                "rag_sources": [],
                "rag_confidence": 0.0,
                "error": str(e),
                "_source": "documentation",
            }
        }


__all__ = ["rag_node"]
