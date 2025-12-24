"""RAG pipeline configuration."""

from pathlib import Path


# =============================================================================
# Paths
# =============================================================================

BACKEND_ROOT = Path(__file__).parent.parent

DOCS_DIR = BACKEND_ROOT / "data/docs"
CSV_DIR = BACKEND_ROOT / "data/csv"
PROCESSED_DIR = BACKEND_ROOT / "data/processed"
QDRANT_PATH = BACKEND_ROOT / "data/qdrant"
AUDIT_LOG_PATH = BACKEND_ROOT / "data/logs/audit.jsonl"
DOC_CHUNKS_PATH = PROCESSED_DIR / "doc_chunks.parquet"


# =============================================================================
# Qdrant Collections
# =============================================================================

DOCS_COLLECTION = "acme_crm_docs"
PRIVATE_COLLECTION = "acme_private_text_v1"
