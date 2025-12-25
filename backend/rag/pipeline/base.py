"""
Shared pipeline utilities and base classes.

Contains common functionality used across different RAG pipelines:
- Progress tracking
- Context building
- Query preprocessing
"""

import logging
import time
from typing import Optional, Callable
from collections import defaultdict

from backend.rag.models import DocumentChunk, ScoredChunk
from backend.rag.pipeline.constants import (
    MAX_CONTEXT_TOKENS,
    MIN_BM25_SCORE_RATIO,
    MAX_CHUNKS_PER_DOC,
    MAX_CHUNKS_PER_TYPE,
)
from backend.rag.pipeline.utils import estimate_tokens, tokens_to_chars

# Re-export gating functions for backwards compatibility
from backend.rag.pipeline.gating import (
    apply_lexical_gate,
    apply_per_doc_cap,
    apply_per_type_cap,
    apply_all_gates,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Progress Tracking
# =============================================================================

class PipelineProgress:
    """
    Tracks and logs pipeline step progress.
    
    Useful for UI progress indicators and debugging.
    """
    
    def __init__(self, callback: Optional[Callable[[str, str, float], None]] = None):
        """
        Initialize progress tracker.
        
        Args:
            callback: Optional function called with (step_id, label, elapsed_ms)
        """
        self.steps: list[dict] = []
        self.callback = callback
        self._start_time = time.time()
        self._step_start: Optional[float] = None
    
    def start_step(self, step_id: str, label: str) -> None:
        """Start tracking a new step."""
        self._step_start = time.time()
        logger.info(f"[STEP] Starting: {label}")
        if self.callback:
            self.callback(step_id, f"Starting: {label}", 0)
    
    def complete_step(self, step_id: str, label: str, status: str = "done") -> None:
        """Mark a step as complete."""
        elapsed_ms = (time.time() - self._step_start) * 1000 if self._step_start else 0
        self.steps.append({
            "id": step_id,
            "label": label,
            "status": status,
            "elapsed_ms": elapsed_ms,
        })
        logger.info(f"[STEP] Completed: {label} ({elapsed_ms:.0f}ms) - {status}")
        if self.callback:
            self.callback(step_id, label, elapsed_ms)
    
    def get_steps(self) -> list[dict]:
        """Get all completed steps."""
        return self.steps
    
    def total_elapsed_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        return (time.time() - self._start_time) * 1000


# =============================================================================
# Context Building
# =============================================================================

def build_context(
    chunks: list[DocumentChunk],
    max_tokens: Optional[int] = None,
) -> str:
    """
    Build a context string from chunks for the LLM prompt.
    
    Args:
        chunks: List of document chunks to include
        max_tokens: Maximum tokens for the context
        
    Returns:
        Formatted context string with doc_id labels
    """
    max_tokens = max_tokens or MAX_CONTEXT_TOKENS
    context_parts = []
    total_tokens = 0
    
    for chunk in chunks:
        # Format: [doc_id] Section: text
        section = chunk.metadata.get("section_heading", "")
        header = f"[{chunk.doc_id}] {section}" if section else f"[{chunk.doc_id}]"
        
        chunk_text = f"{header}\n{chunk.text}\n"
        chunk_tokens = estimate_tokens(chunk_text)
        
        if total_tokens + chunk_tokens > max_tokens:
            # Try to fit partial text
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 50:
                truncated = chunk.text[:tokens_to_chars(remaining_tokens)]
                context_parts.append(f"{header}\n{truncated}...")
            break
        
        context_parts.append(chunk_text)
        total_tokens += chunk_tokens
    
    logger.debug(f"Built context with {len(context_parts)} chunks, ~{total_tokens} tokens")
    return "\n---\n".join(context_parts)


def build_private_context(
    chunks: list[ScoredChunk],
    company_id: str,
    max_tokens: Optional[int] = None,
) -> tuple[str, list[dict]]:
    """
    Build context string from private chunks.
    
    Args:
        chunks: Scored chunks to include
        company_id: The company ID for context header
        max_tokens: Maximum tokens
    
    Returns:
        Tuple of (context_string, sources_list)
    """
    max_tokens = max_tokens or MAX_CONTEXT_TOKENS
    
    # Apply per-type cap
    selected = apply_per_type_cap(chunks)
    
    # Build context
    parts = []
    sources = []
    total_tokens = 0
    
    header = f"PRIVATE CRM CONTEXT (scoped to company_id={company_id})"
    parts.append(header)
    parts.append("=" * 50)
    
    for sc in selected:
        chunk = sc.chunk
        source_id = chunk.metadata.get("source_id", chunk.doc_id)
        chunk_type = chunk.metadata.get("type", "unknown")
        
        # Format chunk
        chunk_text = f"\n[{source_id}] ({chunk_type})\n"
        if chunk.title:
            chunk_text += f"Title: {chunk.title}\n"
        chunk_text += f"{chunk.text}\n"
        
        chunk_tokens = estimate_tokens(chunk_text)
        if total_tokens + chunk_tokens > max_tokens:
            break
        
        parts.append(chunk_text)
        total_tokens += chunk_tokens
        
        sources.append({
            "type": chunk_type,
            "id": source_id,
            "label": chunk.title or source_id,
            "company_id": chunk.metadata.get("company_id", ""),
        })
    
    return "\n".join(parts), sources


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
    
    parts = []
    sources = []
    total_tokens = 0
    
    header = "PRODUCT DOCS CONTEXT"
    parts.append(header)
    parts.append("=" * 50)
    
    for sc in chunks:
        chunk = sc.chunk
        
        chunk_text = f"\n[{chunk.doc_id}]\n{chunk.text}\n"
        chunk_tokens = estimate_tokens(chunk_text)
        
        if total_tokens + chunk_tokens > max_tokens:
            break
        
        parts.append(chunk_text)
        total_tokens += chunk_tokens
        
        sources.append({
            "type": "doc",
            "id": chunk.doc_id,
            "label": chunk.title or chunk.doc_id,
        })
    
    return "\n".join(parts), sources
