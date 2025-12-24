"""RAG pipeline configuration."""

import logging
import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# =============================================================================
# Constants (fixed values, no env override needed)
# =============================================================================

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_DIM = 384

# Qdrant collections
DOCS_COLLECTION = "acme_crm_docs"
PRIVATE_COLLECTION = "acme_private_text_v1"

# Chunking
TARGET_CHUNK_SIZE = 500
MAX_CHUNK_SIZE = 700
MIN_CHUNK_SIZE = 100
CHUNK_OVERLAP = 50
CHARS_PER_TOKEN = 4

# Context building
MAX_CONTEXT_TOKENS = 3000
MAX_CHUNKS_PER_DOC = 3
MAX_CHUNKS_PER_TYPE = 4
MIN_BM25_SCORE_RATIO = 0.1

# Retrieval
DEFAULT_K_DENSE = 20
DEFAULT_K_BM25 = 20
DEFAULT_TOP_N = 10
RRF_K = 60

# LLM
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024
ANSWER_MAX_TOKENS = 800

# Cache
EMBEDDING_CACHE_SIZE = 1000


# =============================================================================
# Dynamic Config (paths - can be overridden via env vars)
# =============================================================================

def _backend_root() -> Path:
    return Path(__file__).parent.parent


class RAGConfig(BaseSettings):
    """RAG paths - override via RAG_ prefix env vars."""

    docs_dir: Path = Field(default_factory=lambda: _backend_root() / "data/docs")
    csv_dir: Path = Field(default_factory=lambda: _backend_root() / "data/csv")
    processed_dir: Path = Field(default_factory=lambda: _backend_root() / "data/processed")
    qdrant_path: Path = Field(default_factory=lambda: _backend_root() / "data/qdrant")
    audit_log_path: Path = Field(default_factory=lambda: _backend_root() / "data/logs/audit.jsonl")

    # Feature flags (may want to toggle in prod)
    enable_query_rewriting: bool = True
    enable_audit_logging: bool = True
    enable_embedding_cache: bool = True

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def doc_chunks_path(self) -> Path:
        return self.processed_dir / "doc_chunks.parquet"

    @property
    def private_texts_path(self) -> Path:
        return self.csv_dir / "private_texts.jsonl"


_config: RAGConfig | None = None


def get_config() -> RAGConfig:
    """Get the global RAG configuration instance."""
    global _config
    if _config is None:
        _config = RAGConfig()
        if not os.environ.get('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY not set")
    return _config


def reset_config() -> None:
    """Reset config (for testing)."""
    global _config
    _config = None
