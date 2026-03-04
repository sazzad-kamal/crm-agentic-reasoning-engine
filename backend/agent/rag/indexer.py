"""PDF document indexer using LlamaIndex.

Loads Act! CRM documentation PDFs and creates a vector index
for semantic search.
"""

import logging
from pathlib import Path
from typing import Optional

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)

# Paths
DOCS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "docs"
INDEX_DIR = Path(__file__).parent.parent.parent.parent / "data" / "index"

# Singleton index instance
_index: Optional[VectorStoreIndex] = None


def _configure_settings() -> None:
    """Configure LlamaIndex global settings."""
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def _load_pdfs() -> list[Document]:
    """Load all PDF documents from the docs directory."""
    from pypdf import PdfReader

    documents = []

    if not DOCS_DIR.exists():
        logger.warning(f"[RAG] Docs directory not found: {DOCS_DIR}")
        return documents

    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    logger.info(f"[RAG] Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")

            if text_parts:
                full_text = "\n\n".join(text_parts)
                doc = Document(
                    text=full_text,
                    metadata={
                        "source": pdf_path.name,
                        "file_path": str(pdf_path),
                        "num_pages": len(reader.pages),
                    },
                )
                documents.append(doc)
                logger.info(f"[RAG] Loaded {pdf_path.name} ({len(reader.pages)} pages)")

        except Exception as e:
            logger.error(f"[RAG] Failed to load {pdf_path.name}: {e}")

    return documents


def build_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """Build or load the vector index.

    Args:
        force_rebuild: If True, rebuild index even if cached version exists

    Returns:
        VectorStoreIndex for semantic search
    """
    global _index

    if _index is not None and not force_rebuild:
        return _index

    _configure_settings()

    # Try to load existing index
    if INDEX_DIR.exists() and not force_rebuild:
        try:
            logger.info("[RAG] Loading existing index from storage")
            storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
            _index = load_index_from_storage(storage_context)
            logger.info("[RAG] Index loaded successfully")
            return _index
        except Exception as e:
            logger.warning(f"[RAG] Failed to load index, rebuilding: {e}")

    # Build new index
    logger.info("[RAG] Building new index from documents")
    documents = _load_pdfs()

    if not documents:
        logger.warning("[RAG] No documents found, creating empty index")
        _index = VectorStoreIndex.from_documents([])
    else:
        _index = VectorStoreIndex.from_documents(documents, show_progress=True)

        # Persist to disk
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _index.storage_context.persist(persist_dir=str(INDEX_DIR))
        logger.info(f"[RAG] Index persisted to {INDEX_DIR}")

    return _index


def get_index() -> VectorStoreIndex:
    """Get the vector index, building if necessary."""
    return build_index()


__all__ = ["build_index", "get_index", "DOCS_DIR", "INDEX_DIR"]
