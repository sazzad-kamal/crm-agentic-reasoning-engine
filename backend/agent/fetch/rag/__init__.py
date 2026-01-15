"""
RAG utilities for the fetch node.

Exports:
    tool_entity_rag: Search entity-scoped CRM text
    ingest_texts: Ingest CRM text into Qdrant
"""

from backend.agent.fetch.rag.search import tool_entity_rag

__all__ = [
    "tool_entity_rag",
]
