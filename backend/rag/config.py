"""RAG pipeline configuration."""

import logging
import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Constants (used across multiple submodules)
# =============================================================================

# Qdrant collections
DOCS_COLLECTION = "acme_crm_docs"
PRIVATE_COLLECTION = "acme_private_text_v1"


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

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def doc_chunks_path(self) -> Path:
        return self.processed_dir / "doc_chunks.parquet"


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
