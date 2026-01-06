"""
RAG search tools for the agent layer.

Provides:
- tool_docs_rag: Search product documentation
- tool_account_rag: Search company-scoped CRM text
"""

import logging

from qdrant_client.models import Filter, FieldCondition, MatchValue

from backend.agent.core.state import Source
from backend.agent.rag.config import PRIVATE_COLLECTION, EMBEDDING_MODEL
from backend.agent.rag.client import get_qdrant_client, get_docs_index


logger = logging.getLogger(__name__)


def _get_embed_model():
    """Get the embedding model (lazy load)."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


def tool_docs_rag(question: str, top_k: int = 5) -> tuple[str, list[Source]]:
    """
    Search product documentation and return relevant context.

    Args:
        question: User's question
        top_k: Number of chunks to retrieve

    Returns:
        Tuple of (context_text, sources)
    """
    try:
        index = get_docs_index()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(question)

        context_parts = []
        sources = []
        seen_docs = set()

        for node in nodes:
            context_parts.append(node.text)
            doc_id = node.metadata.get("doc_id", node.metadata.get("file_name", "unknown"))
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                label = doc_id.replace("_", " ").replace(".md", "").title()
                sources.append(Source(type="doc", id=doc_id, label=label))

        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Docs RAG: retrieved {len(nodes)} chunks from {len(sources)} docs")
        return context, sources

    except Exception as e:
        logger.warning(f"Docs RAG failed: {e}")
        return "", []


def tool_account_rag(
    question: str,
    company_id: str,
    top_k: int = 5,
) -> tuple[str, list[Source]]:
    """
    Search company-scoped CRM text (notes, attachments).

    Args:
        question: User's question
        company_id: Company ID for filtering
        top_k: Number of chunks to retrieve

    Returns:
        Tuple of (context_text, sources)
    """
    try:
        from llama_index.core import VectorStoreIndex, Settings
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

        index = VectorStoreIndex.from_vector_store(vector_store)
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            vector_store_kwargs={"qdrant_filters": qdrant_filter},
        )
        nodes = retriever.retrieve(question)

        context_parts = []
        sources = []

        for node in nodes:
            context_parts.append(node.text)
            source_type = node.metadata.get("type", "note")
            source_id = node.metadata.get("source_id", node.metadata.get("doc_id", "unknown"))
            label = node.metadata.get("title", source_type.replace("_", " ").title())
            sources.append(Source(type=source_type, id=source_id, label=label))

        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Account RAG: retrieved {len(nodes)} chunks for company {company_id}")
        return context, sources

    except Exception as e:
        logger.warning(f"Account RAG failed: {e}")
        return "", []


__all__ = [
    "tool_docs_rag",
    "tool_account_rag",
]
