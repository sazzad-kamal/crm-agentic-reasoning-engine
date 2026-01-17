"""
RAG search tools for the agent layer.

Provides:
- search_entity_context: Search entity-scoped CRM text
"""

import logging
import threading

from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.agent.fetch.rag.client import get_qdrant_client
from backend.agent.fetch.rag.config import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANKER_TOP_K,
    SEARCH_TOP_K,
    SPARSE_EMBEDDING_MODEL,
    TEXT_COLLECTION,
)

logger = logging.getLogger(__name__)

# Thread-safe lazy initialization
_embed_model = None
_vector_index = None
_reranker = None
_init_lock = threading.Lock()


def _ensure_initialized():
    """Initialize embedding model, vector index, and reranker (thread-safe)."""
    global _embed_model, _vector_index, _reranker
    if _vector_index is not None:
        return

    with _init_lock:
        if _vector_index is not None:
            return

        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.postprocessor import SentenceTransformerRerank
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        # Initialize embedding model
        _embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
        Settings.embed_model = _embed_model

        # Build vector store with hybrid search
        client = get_qdrant_client()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=TEXT_COLLECTION,
            enable_hybrid=True,
            fastembed_sparse_model=SPARSE_EMBEDDING_MODEL,
        )
        _vector_index = VectorStoreIndex.from_vector_store(vector_store)

        # Initialize reranker
        _reranker = SentenceTransformerRerank(model=RERANKER_MODEL, top_n=RERANKER_TOP_K)

        logger.debug("Initialized RAG components")


def search_entity_context(
    question: str,
    filters: dict[str, str],
) -> tuple[str, list[dict]]:
    """
    Search entity-scoped CRM text (notes, descriptions).

    Uses over-retrieval + reranking for better precision.
    Filters by ANY provided entity ID (OR logic) to get all related documents.

    Args:
        question: User's question
        filters: Dict of entity IDs to filter by (company_id, contact_id, opportunity_id, activity_id)

    Returns:
        Tuple of (context_text, source_metadata)
    """
    try:
        _ensure_initialized()

        # Build OR filter - match documents with ANY of the provided entity IDs
        should_conditions = []
        for key, value in filters.items():
            if value:
                should_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        qdrant_filter = Filter(should=should_conditions) if should_conditions else None  # type: ignore[arg-type]

        # Configure hybrid retriever (dense + sparse)
        retriever = _vector_index.as_retriever(  # type: ignore[union-attr]
            similarity_top_k=SEARCH_TOP_K,
            sparse_top_k=SEARCH_TOP_K,
            vector_store_query_mode="hybrid",
            vector_store_kwargs={"qdrant_filters": qdrant_filter},
        )
        nodes = retriever.retrieve(question)

        # Rerank to top-k
        if len(nodes) > RERANKER_TOP_K:
            from llama_index.core.schema import QueryBundle

            nodes = _reranker.postprocess_nodes(nodes, QueryBundle(query_str=question))  # type: ignore[union-attr]
        logger.info(f"Entity RAG: {len(nodes)} chunks with filters={filters}")

        context_parts = []
        sources = []

        for node in nodes:
            context_parts.append(node.text)
            source_type = node.metadata.get("type", "note")
            source_id = node.metadata.get("source_id", node.metadata.get("doc_id", "unknown"))
            sources.append({"type": source_type, "id": source_id})

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    except Exception as e:
        logger.warning(f"Entity RAG failed: {e}")
        return "", []


__all__ = [
    "search_entity_context",
]
