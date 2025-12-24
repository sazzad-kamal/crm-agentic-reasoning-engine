"""
Document chunk models for the RAG pipeline.
"""

from pydantic import BaseModel, Field

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
    metadata: dict = Field(default_factory=dict)

class ScoredChunk(BaseModel):
    """
    A document chunk with retrieval scores attached.
    """
    chunk: DocumentChunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
