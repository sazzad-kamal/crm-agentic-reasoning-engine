"""RAG pipeline configuration."""

from pathlib import Path


# =============================================================================
# Paths
# =============================================================================

BACKEND_ROOT = Path(__file__).parent.parent

DOCS_DIR = BACKEND_ROOT / "data/docs"
QDRANT_PATH = BACKEND_ROOT / "data/qdrant"
DOC_CHUNKS_PATH = BACKEND_ROOT / "data/processed/doc_chunks.parquet"


# =============================================================================
# Qdrant Collections
# =============================================================================

PRIVATE_COLLECTION = "acme_private_text_v1"
