"""
LlamaIndex-based RAG tools for the agent layer.

Provides:
- tool_docs_rag: Search product documentation
- tool_account_rag: Search company-scoped CRM text
- ingest_docs: Ingest markdown docs into Qdrant
- ingest_private_texts: Ingest CRM private text into Qdrant

Uses Qdrant for vector storage with basic similarity search.
"""

import json
import logging
import threading
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from backend.agent.schemas import Source

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

_BACKEND_ROOT = Path(__file__).parent.parent
QDRANT_PATH = _BACKEND_ROOT / "data" / "qdrant"
DOCS_COLLECTION = "acme_crm_docs"
PRIVATE_COLLECTION = "acme_crm_private"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Ingestion paths
DOCS_DIR = _BACKEND_ROOT / "data" / "docs"
JSONL_PATH = _BACKEND_ROOT / "data" / "csv" / "private_texts.jsonl"

# =============================================================================
# Singleton Qdrant Client
# =============================================================================

_qdrant_client: QdrantClient | None = None
_qdrant_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    """Get shared Qdrant client (singleton)."""
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    with _qdrant_lock:
        if _qdrant_client is not None:
            return _qdrant_client

        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(QDRANT_PATH))
        return _qdrant_client


def close_qdrant_client() -> None:
    """Close the shared Qdrant client (for cleanup)."""
    global _qdrant_client
    with _qdrant_lock:
        if _qdrant_client is not None:
            _qdrant_client.close()
            _qdrant_client = None


# =============================================================================
# LlamaIndex Index Singletons
# =============================================================================

_docs_index = None
_private_index = None
_index_lock = threading.Lock()
_embed_model = None


def _get_embed_model():
    """Get the embedding model (lazy load)."""
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    return _embed_model


def get_docs_index():
    """Get or create the docs vector index."""
    global _docs_index

    if _docs_index is not None:
        return _docs_index

    with _index_lock:
        if _docs_index is not None:
            return _docs_index

        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        Settings.embed_model = _get_embed_model()

        client = get_qdrant_client()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=DOCS_COLLECTION,
        )
        _docs_index = VectorStoreIndex.from_vector_store(vector_store)
        return _docs_index


def get_private_index():
    """Get or create the private text vector index."""
    global _private_index

    if _private_index is not None:
        return _private_index

    with _index_lock:
        if _private_index is not None:
            return _private_index

        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        Settings.embed_model = _get_embed_model()

        client = get_qdrant_client()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=PRIVATE_COLLECTION,
        )
        _private_index = VectorStoreIndex.from_vector_store(vector_store)
        return _private_index


# =============================================================================
# RAG Tools
# =============================================================================

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


# =============================================================================
# Collection Existence Check
# =============================================================================

def collections_exist() -> tuple[bool, bool]:
    """
    Check if RAG collections exist in Qdrant.

    Returns:
        Tuple of (docs_exists, private_exists)
    """
    try:
        client = get_qdrant_client()
        docs_exists = client.collection_exists(DOCS_COLLECTION)
        private_exists = client.collection_exists(PRIVATE_COLLECTION)
        return docs_exists, private_exists
    except Exception as e:
        logger.warning(f"Error checking collections: {e}")
        return False, False


# =============================================================================
# Ingestion Functions
# =============================================================================

def ingest_docs(recreate: bool = True) -> int:
    """
    Ingest all markdown docs into Qdrant.

    Args:
        recreate: If True, delete and recreate the collection

    Returns:
        Number of chunks ingested
    """
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    logger.info(f"Ingesting docs from {DOCS_DIR}")

    # Configure LlamaIndex
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Load markdown documents
    if not DOCS_DIR.exists():
        logger.warning(f"Docs directory not found: {DOCS_DIR}")
        return 0

    reader = SimpleDirectoryReader(
        input_dir=str(DOCS_DIR),
        required_exts=[".md"],
        recursive=False,
    )
    documents = reader.load_data()

    if not documents:
        logger.warning("No markdown documents found")
        return 0

    # Add doc_id metadata from filename
    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        doc_id = Path(file_path).stem if file_path else "unknown"
        doc.metadata["doc_id"] = doc_id

    logger.info(f"Loaded {len(documents)} documents")

    # Initialize Qdrant
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    # Delete existing collection if recreate=True
    if recreate and client.collection_exists(DOCS_COLLECTION):
        logger.info(f"Deleting existing collection: {DOCS_COLLECTION}")
        client.delete_collection(DOCS_COLLECTION)

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=DOCS_COLLECTION,
    )

    # Build index (this ingests the documents)
    logger.info("Building vector index...")
    VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        show_progress=True,
    )

    # Get final count
    info = client.get_collection(DOCS_COLLECTION)
    chunk_count = info.points_count

    logger.info(f"Ingested {chunk_count} chunks into '{DOCS_COLLECTION}'")

    # Close the client to release the lock
    client.close()

    return chunk_count


def ingest_private_texts(recreate: bool = True) -> int:
    """
    Ingest private texts from JSONL into Qdrant.

    Args:
        recreate: If True, delete and recreate the collection

    Returns:
        Number of chunks ingested
    """
    from llama_index.core import Document, VectorStoreIndex, Settings
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
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
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

    # Initialize Qdrant
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    # Delete existing collection if recreate=True
    if recreate and client.collection_exists(PRIVATE_COLLECTION):
        logger.info(f"Deleting existing collection: {PRIVATE_COLLECTION}")
        client.delete_collection(PRIVATE_COLLECTION)

    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=PRIVATE_COLLECTION,
    )

    # Build index (this ingests the documents)
    logger.info("Building vector index...")
    VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        show_progress=True,
    )

    # Get final count
    info = client.get_collection(PRIVATE_COLLECTION)
    chunk_count = info.points_count

    logger.info(f"Ingested {chunk_count} chunks into '{PRIVATE_COLLECTION}'")

    # Close the client to release the lock
    client.close()

    return chunk_count


__all__ = [
    "tool_docs_rag",
    "tool_account_rag",
    "collections_exist",
    "get_qdrant_client",
    "close_qdrant_client",
    "ingest_docs",
    "ingest_private_texts",
    "QDRANT_PATH",
    "DOCS_COLLECTION",
    "PRIVATE_COLLECTION",
    "EMBEDDING_MODEL",
]
