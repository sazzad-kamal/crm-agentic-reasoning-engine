"""
RAG utilities for the fetch node.

Exports:
    search_entity_context: Search entity-scoped CRM text
    ingest_texts: Ingest CRM text into Qdrant
"""

from backend.agent.fetch.rag.search import search_entity_context

__all__ = [
    "search_entity_context",
]
