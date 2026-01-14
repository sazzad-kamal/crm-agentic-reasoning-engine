"""
RAG ingestion functions.

Provides:
- ingest_private_texts: Ingest CRM private text into Qdrant
"""

import json
import logging

from qdrant_client import QdrantClient

from backend.agent.fetch.rag.client import close_qdrant_client
from backend.agent.fetch.rag.config import (
    EMBEDDING_MODEL,
    HYBRID_SEARCH_ENABLED,
    JSONL_PATH,
    PRIVATE_COLLECTION,
    QDRANT_PATH,
    SPARSE_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


def ingest_private_texts(recreate: bool = True) -> int:  # pragma: no cover
    """
    Ingest private texts from JSONL into Qdrant.

    Requires external dependencies (Qdrant, llama_index, HuggingFace).
    Run via: python -m backend.agent.fetch.rag.ingest

    Args:
        recreate: If True, delete and recreate the collection

    Returns:
        Number of chunks ingested
    """
    from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    logger.info(f"Ingesting private texts from {JSONL_PATH}")

    # Configure LlamaIndex
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Load JSONL documents
    if not JSONL_PATH.exists():
        logger.warning(f"Private texts file not found: {JSONL_PATH}")
        return 0

    documents = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                doc = Document(
                    text=record.get("text", ""),
                    metadata={
                        "doc_id": record.get("id", ""),
                        "source_id": record.get("id", ""),
                        "company_id": record.get("company_id", ""),
                        "type": record.get("type", ""),
                        "title": record.get("title", ""),
                        "contact_id": record.get("contact_id"),
                        "opportunity_id": record.get("opportunity_id"),
                    },
                )
                documents.append(doc)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

    if not documents:
        logger.warning("No documents loaded from JSONL")
        return 0

    logger.info(f"Loaded {len(documents)} documents from JSONL")

    # Close any existing singleton to avoid conflicts
    close_qdrant_client()

    # Create dedicated client for ingestion (local Qdrant needs exclusive access)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    # Delete existing collection if recreate=True
    if recreate and client.collection_exists(PRIVATE_COLLECTION):
        logger.info(f"Deleting existing collection: {PRIVATE_COLLECTION}")
        client.delete_collection(PRIVATE_COLLECTION)

    # Create vector store with storage context
    vector_store_kwargs = {
        "client": client,
        "collection_name": PRIVATE_COLLECTION,
    }
    if HYBRID_SEARCH_ENABLED:
        vector_store_kwargs["enable_hybrid"] = True
        vector_store_kwargs["fastembed_sparse_model"] = SPARSE_EMBEDDING_MODEL
        logger.info(f"Hybrid search enabled with sparse model: {SPARSE_EMBEDDING_MODEL}")

    vector_store = QdrantVectorStore(**vector_store_kwargs)  # type: ignore[arg-type]
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index (this ingests the documents)
    logger.info("Building vector index...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    # Persist to ensure data is written
    storage_context.persist()

    chunk_count = len(documents)
    logger.info(f"Ingested {chunk_count} documents into '{PRIVATE_COLLECTION}'")

    # Close client to persist data (singleton will be recreated on next access)
    client.close()
    return chunk_count


__all__ = [
    "ingest_private_texts",
]
