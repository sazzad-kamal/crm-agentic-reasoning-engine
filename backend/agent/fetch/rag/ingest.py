"""
RAG ingestion functions.

Provides:
- ingest_texts: Ingest CRM text into Qdrant
"""

import json
import logging

from qdrant_client import QdrantClient

from backend.agent.fetch.rag.client import close_qdrant_client
from backend.agent.fetch.rag.config import (
    EMBEDDING_MODEL,
    JSONL_PATH,
    QDRANT_PATH,
    SPARSE_EMBEDDING_MODEL,
    TEXT_COLLECTION,
)

logger = logging.getLogger(__name__)


def ingest_texts(recreate: bool = True) -> int:  # pragma: no cover
    """
    Ingest texts from JSONL into Qdrant.

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

    logger.info(f"Ingesting texts from {JSONL_PATH}")

    # Configure LlamaIndex (disable LLM since we only need embeddings)
    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Load JSONL documents
    if not JSONL_PATH.exists():
        logger.warning(f"Texts file not found: {JSONL_PATH}")
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
                        "company_id": record.get("company_id", ""),
                        "type": record.get("type", ""),
                        "contact_id": record.get("contact_id"),
                        "opportunity_id": record.get("opportunity_id"),
                        "activity_id": record.get("activity_id"),
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

    try:
        # Delete existing collection if recreate=True
        if recreate and client.collection_exists(TEXT_COLLECTION):
            logger.info(f"Deleting existing collection: {TEXT_COLLECTION}")
            client.delete_collection(TEXT_COLLECTION)

        # Create vector store with hybrid search (dense + sparse)
        vector_store_kwargs = {
            "client": client,
            "collection_name": TEXT_COLLECTION,
            "enable_hybrid": True,
            "fastembed_sparse_model": SPARSE_EMBEDDING_MODEL,
        }
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

        # Get actual chunk count from Qdrant (documents are split into multiple chunks)
        chunk_count = client.count(collection_name=TEXT_COLLECTION).count
        logger.info(f"Ingested {chunk_count} chunks into '{TEXT_COLLECTION}'")

        return chunk_count
    finally:
        client.close()


__all__ = [
    "ingest_texts",
]
