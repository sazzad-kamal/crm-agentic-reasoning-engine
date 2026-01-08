"""
RAG search tools for the agent layer.

Provides:
- tool_account_rag: Search company-scoped CRM text
"""

import logging

from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.agent.core.state import Source
from backend.agent.rag.client import get_qdrant_client
from backend.agent.rag.config import (
    EMBEDDING_MODEL,
    PRIVATE_COLLECTION,
    RERANKER_ENABLED,
    RERANKER_TOP_K,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)


def _get_embed_model():
    """Get the embedding model (lazy load)."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


def tool_account_rag(
    question: str,
    company_id: str,
    top_k: int = 5,
) -> tuple[str, list[Source]]:
    """
    Search company-scoped CRM text (notes, attachments).

    Uses over-retrieval + reranking for better precision when enabled.

    Args:
        question: User's question
        company_id: Company ID for filtering
        top_k: Number of chunks to return (after reranking if enabled)

    Returns:
        Tuple of (context_text, sources)
    """
    try:
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        Settings.embed_model = _get_embed_model()

        client = get_qdrant_client()

        # Create vector store with Qdrant filter for company_id
        qdrant_filter = Filter(
            must=[FieldCondition(key="company_id", match=MatchValue(value=company_id))]
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=PRIVATE_COLLECTION,
        )

        # Over-retrieve if reranker is enabled, otherwise use top_k directly
        retrieval_k = RETRIEVAL_TOP_K if RERANKER_ENABLED else top_k

        index = VectorStoreIndex.from_vector_store(vector_store)
        retriever = index.as_retriever(
            similarity_top_k=retrieval_k,
            vector_store_kwargs={"qdrant_filters": qdrant_filter},
        )
        nodes = retriever.retrieve(question)

        # Rerank if enabled and we have more nodes than needed
        if RERANKER_ENABLED and len(nodes) > RERANKER_TOP_K:
            from backend.agent.rag.reranker import rerank_nodes

            nodes = rerank_nodes(nodes, question, top_k=RERANKER_TOP_K)
            logger.info(f"Account RAG: reranked to {len(nodes)} chunks for company {company_id}")
        else:
            logger.info(f"Account RAG: retrieved {len(nodes)} chunks for company {company_id}")

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
        logger.warning(f"Account RAG failed: {e}")
        return "", []


__all__ = [
    "tool_account_rag",
]
