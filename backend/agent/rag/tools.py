"""
RAG search tools for the agent layer.

Provides:
- tool_entity_rag: Search entity-scoped CRM text
"""

import logging
import threading

from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.agent.core.state import Source
from backend.agent.rag.client import get_qdrant_client
from backend.agent.rag.config import (
    EMBEDDING_MODEL,
    HYBRID_SEARCH_ENABLED,
    PRIVATE_COLLECTION,
    RERANKER_ENABLED,
    RERANKER_TOP_K,
    RETRIEVAL_TOP_K,
    SPARSE_EMBEDDING_MODEL,
    SPARSE_TOP_K,
)

logger = logging.getLogger(__name__)

# Thread-safe lazy initialization for embedding model
_embed_model = None
_embed_model_lock = threading.Lock()


def _get_embed_model():
    """Get the embedding model (thread-safe lazy initialization)."""
    global _embed_model
    if _embed_model is None:
        with _embed_model_lock:
            # Double-check after acquiring lock
            if _embed_model is None:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                _embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    return _embed_model


def tool_entity_rag(
    question: str,
    filters: dict[str, str],
    top_k: int = 5,
) -> tuple[str, list[Source]]:
    """
    Search entity-scoped CRM text (notes, attachments).

    Uses over-retrieval + reranking for better precision when enabled.
    Filters by all provided entity IDs for precise results.

    Args:
        question: User's question
        filters: Dict of entity IDs to filter by (company_id, contact_id, opportunity_id)
        top_k: Number of chunks to return (after reranking if enabled)

    Returns:
        Tuple of (context_text, sources)
    """
    try:
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        Settings.embed_model = _get_embed_model()

        client = get_qdrant_client()

        # Build compound filter from all provided entity IDs
        must_conditions = []
        for key, value in filters.items():
            if value:
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        # Configure vector store with hybrid if enabled
        vector_store_kwargs = {
            "client": client,
            "collection_name": PRIVATE_COLLECTION,
        }
        if HYBRID_SEARCH_ENABLED:
            vector_store_kwargs["enable_hybrid"] = True
            vector_store_kwargs["fastembed_sparse_model"] = SPARSE_EMBEDDING_MODEL

        vector_store = QdrantVectorStore(**vector_store_kwargs)

        # Over-retrieve if reranker is enabled, otherwise use top_k directly
        retrieval_k = RETRIEVAL_TOP_K if RERANKER_ENABLED else top_k

        index = VectorStoreIndex.from_vector_store(vector_store)

        # Configure retriever with hybrid mode if enabled
        retriever_kwargs = {
            "similarity_top_k": retrieval_k,
            "vector_store_kwargs": {"qdrant_filters": qdrant_filter},
        }
        if HYBRID_SEARCH_ENABLED:
            retriever_kwargs["sparse_top_k"] = SPARSE_TOP_K
            retriever_kwargs["vector_store_query_mode"] = "hybrid"

        retriever = index.as_retriever(**retriever_kwargs)
        nodes = retriever.retrieve(question)

        # Rerank if enabled and we have more nodes than needed
        if RERANKER_ENABLED and len(nodes) > RERANKER_TOP_K:
            from backend.agent.rag.reranker import rerank_nodes

            nodes = rerank_nodes(nodes, question, top_k=RERANKER_TOP_K)
            logger.info(f"Entity RAG: reranked to {len(nodes)} chunks with filters={filters}")
        else:
            logger.info(f"Entity RAG: retrieved {len(nodes)} chunks with filters={filters}")

        context_parts = []
        sources = []

        for node in nodes:
            context_parts.append(node.text)
            source_type = node.metadata.get("type", "note")
            source_id = node.metadata.get("source_id", node.metadata.get("doc_id", "unknown"))
            label = node.metadata.get("title", source_type.replace("_", " ").title())
            sources.append(Source(type=source_type, id=source_id, label=label))

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    except Exception as e:
        logger.warning(f"Entity RAG failed: {e}")
        return "", []


__all__ = [
    "tool_entity_rag",
]
