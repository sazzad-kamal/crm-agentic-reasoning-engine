"""
Fetch node RAG wrappers.

Thin wrappers around RAG tools for error handling.
"""

import logging

from backend.agent.core.state import Source

logger = logging.getLogger(__name__)


def call_account_rag(
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


__all__ = [
    "call_account_rag",
]
