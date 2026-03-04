"""RAG (Retrieval Augmented Generation) agent for Act! CRM documentation.

This agent handles questions about how to use Act! CRM by searching
through official documentation using semantic similarity.
"""

from backend.agent.rag.node import rag_node

__all__ = ["rag_node"]
