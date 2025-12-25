"""
Unified context building utilities.

Consolidates context building logic previously duplicated across:
- backend/rag/pipeline/base.py (build_context, build_private_context, build_docs_context)
- backend/rag/pipeline/account.py

Provides a single, flexible context builder that handles all chunk types.
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.rag.models import DocumentChunk, ScoredChunk


logger = logging.getLogger(__name__)


# Default token estimation (4 chars per token approximation)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def tokens_to_chars(tokens: int) -> int:
    """Convert token count to approximate character count."""
    return tokens * CHARS_PER_TOKEN


class ContextBuilder:
    """
    Unified context builder for RAG pipelines.
    
    Handles:
    - Token counting and truncation
    - Source tracking
    - Flexible formatting for different chunk types
    - Headers and separators
    """
    
    def __init__(
        self,
        max_tokens: int = 3000,
        header: Optional[str] = None,
        separator: str = "\n---\n",
        token_estimator: Optional[Callable[[str], int]] = None,
    ):
        """
        Initialize the context builder.
        
        Args:
            max_tokens: Maximum tokens for the context
            header: Optional header text to prepend
            separator: Separator between chunks
            token_estimator: Custom token estimation function
        """
        self.max_tokens = max_tokens
        self.header = header
        self.separator = separator
        self.estimate_tokens = token_estimator or estimate_tokens
        
        self._parts: list[str] = []
        self._sources: list[dict] = []
        self._total_tokens = 0
        
        # Add header if provided
        if header:
            self._add_header(header)
    
    def _add_header(self, header: str) -> None:
        """Add a header to the context."""
        header_text = f"{header}\n{'=' * 50}\n"
        self._parts.append(header_text)
        self._total_tokens += self.estimate_tokens(header_text)
    
    def add_chunk(
        self,
        chunk: DocumentChunk,
        score: Optional[float] = None,
        format_fn: Optional[Callable[[DocumentChunk, Optional[float]], str]] = None,
    ) -> bool:
        """
        Add a document chunk to the context.
        
        Args:
            chunk: The document chunk to add
            score: Optional relevance score
            format_fn: Optional custom formatting function
            
        Returns:
            True if chunk was added, False if it would exceed max_tokens
        """
        if format_fn:
            chunk_text = format_fn(chunk, score)
        else:
            chunk_text = self._default_format(chunk)
        
        chunk_tokens = self.estimate_tokens(chunk_text)
        
        if self._total_tokens + chunk_tokens > self.max_tokens:
            # Try to fit partial text
            remaining_tokens = self.max_tokens - self._total_tokens
            if remaining_tokens > 50:
                truncated = chunk.text[:tokens_to_chars(remaining_tokens)]
                self._parts.append(f"[{chunk.doc_id}]\n{truncated}...")
                self._add_source(chunk)
            return False
        
        self._parts.append(chunk_text)
        self._total_tokens += chunk_tokens
        self._add_source(chunk)
        return True
    
    def add_scored_chunk(
        self,
        scored_chunk: ScoredChunk,
        format_fn: Optional[Callable[[DocumentChunk, Optional[float]], str]] = None,
    ) -> bool:
        """Add a scored chunk to the context."""
        return self.add_chunk(scored_chunk.chunk, scored_chunk.score, format_fn)
    
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        format_fn: Optional[Callable[[DocumentChunk, Optional[float]], str]] = None,
    ) -> int:
        """
        Add multiple chunks to the context.
        
        Returns the number of chunks added.
        """
        added = 0
        for chunk in chunks:
            if self.add_chunk(chunk, format_fn=format_fn):
                added += 1
            else:
                break
        return added
    
    def add_scored_chunks(
        self,
        scored_chunks: list[ScoredChunk],
        format_fn: Optional[Callable[[DocumentChunk, Optional[float]], str]] = None,
    ) -> int:
        """
        Add multiple scored chunks to the context.
        
        Returns the number of chunks added.
        """
        added = 0
        for sc in scored_chunks:
            if self.add_scored_chunk(sc, format_fn=format_fn):
                added += 1
            else:
                break
        return added
    
    def _default_format(self, chunk: DocumentChunk) -> str:
        """Default formatting for a chunk."""
        section = chunk.metadata.get("section_heading", "")
        header = f"[{chunk.doc_id}] {section}" if section else f"[{chunk.doc_id}]"
        
        lines = [header]
        if chunk.title and chunk.title != chunk.doc_id:
            lines.append(f"Title: {chunk.title}")
        lines.append(chunk.text)
        lines.append("")  # Trailing newline
        
        return "\n".join(lines)
    
    def _add_source(self, chunk: DocumentChunk) -> None:
        """Track a source from a chunk."""
        source_id = chunk.metadata.get("source_id", chunk.doc_id)
        chunk_type = chunk.metadata.get("type", "doc")
        
        self._sources.append({
            "type": chunk_type,
            "id": source_id,
            "label": chunk.title or source_id,
            "doc_id": chunk.doc_id,
            "company_id": chunk.metadata.get("company_id", ""),
        })
    
    def build(self) -> str:
        """Build and return the context string."""
        return self.separator.join(self._parts)
    
    def get_sources(self) -> list[dict]:
        """Get the list of sources included in the context."""
        return self._sources.copy()
    
    def get_unique_doc_ids(self) -> list[str]:
        """Get unique doc_ids from sources."""
        seen = set()
        result = []
        for s in self._sources:
            doc_id = s.get("doc_id", s.get("id"))
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                result.append(doc_id)
        return result
    
    @property
    def total_tokens(self) -> int:
        """Get the current total token count."""
        return self._total_tokens
    
    @property
    def remaining_tokens(self) -> int:
        """Get remaining token budget."""
        return max(0, self.max_tokens - self._total_tokens)


# =============================================================================
# Convenience functions for common patterns
# =============================================================================

def build_context(
    chunks: list[DocumentChunk],
    max_tokens: int = 3000,
) -> str:
    """
    Build a context string from document chunks.
    
    Simple convenience function for basic use cases.
    """
    builder = ContextBuilder(max_tokens=max_tokens)
    builder.add_chunks(chunks)
    return builder.build()


def build_context_with_sources(
    chunks: list[DocumentChunk],
    max_tokens: int = 3000,
) -> tuple[str, list[dict]]:
    """
    Build context string and return sources.
    
    Returns:
        Tuple of (context_string, sources_list)
    """
    builder = ContextBuilder(max_tokens=max_tokens)
    builder.add_chunks(chunks)
    return builder.build(), builder.get_sources()


def build_private_context(
    chunks: list[ScoredChunk],
    company_id: str,
    max_tokens: int = 3000,
) -> tuple[str, list[dict]]:
    """
    Build context string from private CRM chunks.
    
    Args:
        chunks: Scored chunks to include
        company_id: The company ID for context header
        max_tokens: Maximum tokens
    
    Returns:
        Tuple of (context_string, sources_list)
    """
    def format_private_chunk(chunk: DocumentChunk, score: Optional[float]) -> str:
        source_id = chunk.metadata.get("source_id", chunk.doc_id)
        chunk_type = chunk.metadata.get("type", "unknown")
        
        lines = [f"[{source_id}] ({chunk_type})"]
        if chunk.title:
            lines.append(f"Title: {chunk.title}")
        lines.append(chunk.text)
        lines.append("")
        return "\n".join(lines)
    
    builder = ContextBuilder(
        max_tokens=max_tokens,
        header=f"PRIVATE CRM CONTEXT (scoped to company_id={company_id})",
    )
    builder.add_scored_chunks(chunks, format_fn=format_private_chunk)
    return builder.build(), builder.get_sources()


def build_docs_context(
    chunks: list[ScoredChunk],
    max_tokens: int = 1000,
) -> tuple[str, list[dict]]:
    """
    Build context string from product docs chunks.
    
    Returns:
        Tuple of (context_string, sources_list)
    """
    if not chunks:
        return "", []
    
    def format_doc_chunk(chunk: DocumentChunk, score: Optional[float]) -> str:
        return f"[{chunk.doc_id}]\n{chunk.text}\n"
    
    builder = ContextBuilder(
        max_tokens=max_tokens,
        header="PRODUCT DOCS CONTEXT",
    )
    builder.add_scored_chunks(chunks, format_fn=format_doc_chunk)
    return builder.build(), builder.get_sources()
