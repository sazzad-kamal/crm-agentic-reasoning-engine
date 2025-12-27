"""
Shared data models used across backend modules.

These models are foundational and used by both backend.rag and backend.common,
so they live in backend.common to avoid circular dependencies.
"""

from typing import Any
from pydantic import BaseModel, Field


__all__ = [
    "DocumentChunk",
    "ScoredChunk",
]


class DocumentChunk(BaseModel):
    """
    Represents a single chunk of a document for retrieval.
    
    Attributes:
        chunk_id: Unique identifier for the chunk (e.g., "doc_id::chunk_index")
        doc_id: The source document identifier (filename without extension)
        title: The document or section title
        text: The actual text content of the chunk
        metadata: Additional metadata (section headings, file name, etc.)
    """
    chunk_id: str
    doc_id: str
    title: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """
    A document chunk with retrieval scores attached.
    """
    chunk: DocumentChunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
