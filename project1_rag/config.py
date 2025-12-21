"""
Centralized configuration for the RAG pipeline.

All constants, model names, paths, and tunable parameters are defined here.
Import from this module instead of defining constants in individual files.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    """
    RAG pipeline configuration with environment variable support.
    
    All settings can be overridden via environment variables with the
    RAG_ prefix (e.g., RAG_MAX_CONTEXT_TOKENS=4000).
    """
    
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    docs_dir: Path = Field(default=Path("data/docs"))
    csv_dir: Path = Field(default=Path("data/csv"))
    processed_dir: Path = Field(default=Path("data/processed"))
    qdrant_path: Path = Field(default=Path("data/qdrant"))
    
    # Output files
    doc_chunks_file: str = "doc_chunks.parquet"
    private_texts_file: str = "private_texts.jsonl"
    
    # -------------------------------------------------------------------------
    # Model Names
    # -------------------------------------------------------------------------
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Sentence transformer model for embeddings"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    llm_model: str = Field(
        default="gpt-4.1-mini",
        description="LLM model for generation"
    )
    
    # Embedding dimension (must match embedding_model)
    embedding_dim: int = 384
    
    # -------------------------------------------------------------------------
    # Qdrant Collections
    # -------------------------------------------------------------------------
    docs_collection_name: str = "acme_crm_docs"
    private_collection_name: str = "acme_private_text_v1"
    
    # -------------------------------------------------------------------------
    # Chunking Parameters
    # -------------------------------------------------------------------------
    target_chunk_size: int = Field(
        default=500,
        description="Target chunk size in tokens"
    )
    max_chunk_size: int = Field(
        default=700,
        description="Maximum chunk size in tokens"
    )
    min_chunk_size: int = Field(
        default=100,
        description="Minimum chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks in tokens"
    )
    chars_per_token: int = Field(
        default=4,
        description="Approximate characters per token for estimation"
    )
    
    # -------------------------------------------------------------------------
    # Context Building
    # -------------------------------------------------------------------------
    max_context_tokens: int = Field(
        default=3000,
        description="Maximum tokens for LLM context"
    )
    max_chunks_per_doc: int = Field(
        default=3,
        description="Maximum chunks per document in context"
    )
    max_chunks_per_type: int = Field(
        default=4,
        description="Maximum chunks per type (for private retrieval)"
    )
    min_bm25_score_ratio: float = Field(
        default=0.1,
        description="Minimum BM25 score as ratio of top score"
    )
    
    # -------------------------------------------------------------------------
    # Retrieval Parameters
    # -------------------------------------------------------------------------
    default_k_dense: int = Field(
        default=20,
        description="Default number of results from dense search"
    )
    default_k_bm25: int = Field(
        default=20,
        description="Default number of results from BM25 search"
    )
    default_top_n: int = Field(
        default=10,
        description="Default number of final results after reranking"
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant for score fusion"
    )
    
    # -------------------------------------------------------------------------
    # LLM Parameters
    # -------------------------------------------------------------------------
    llm_temperature: float = Field(
        default=0.0,
        description="LLM temperature (0.0 for deterministic)"
    )
    llm_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens in LLM response"
    )
    answer_max_tokens: int = Field(
        default=800,
        description="Maximum tokens for answer generation"
    )
    
    # -------------------------------------------------------------------------
    # Retry Configuration
    # -------------------------------------------------------------------------
    llm_max_retries: int = Field(
        default=3,
        description="Maximum retries for LLM calls"
    )
    llm_retry_min_wait: float = Field(
        default=1.0,
        description="Minimum wait time between retries (seconds)"
    )
    llm_retry_max_wait: float = Field(
        default=10.0,
        description="Maximum wait time between retries (seconds)"
    )
    
    # -------------------------------------------------------------------------
    # Cache Configuration
    # -------------------------------------------------------------------------
    enable_embedding_cache: bool = Field(
        default=True,
        description="Enable caching for embeddings"
    )
    embedding_cache_size: int = Field(
        default=1000,
        description="Maximum entries in embedding cache"
    )
    enable_llm_cache: bool = Field(
        default=False,
        description="Enable caching for LLM responses (use with caution)"
    )
    llm_cache_size: int = Field(
        default=100,
        description="Maximum entries in LLM cache"
    )
    
    # -------------------------------------------------------------------------
    # Cost Estimation (GPT-4.1-mini pricing)
    # -------------------------------------------------------------------------
    cost_per_input_token: float = 0.40 / 1_000_000
    cost_per_output_token: float = 1.60 / 1_000_000
    
    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        extra = "ignore"
    
    @property
    def doc_chunks_path(self) -> Path:
        """Full path to doc chunks file."""
        return self.processed_dir / self.doc_chunks_file
    
    @property
    def private_texts_path(self) -> Path:
        """Full path to private texts file."""
        return self.csv_dir / self.private_texts_file


# Global config instance (lazy initialization)
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get the global RAG configuration instance."""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None


# Convenience exports for common settings
def get_chars_per_token() -> int:
    return get_config().chars_per_token


def get_embedding_model() -> str:
    return get_config().embedding_model


def get_reranker_model() -> str:
    return get_config().reranker_model


def get_qdrant_path() -> Path:
    return get_config().qdrant_path
