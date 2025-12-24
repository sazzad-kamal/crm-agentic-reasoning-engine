"""
Shared pipeline utilities and base classes.

Contains common functionality used across different RAG pipelines:
- Progress tracking
- Context building
- Query preprocessing
- Gating and filtering
"""

import logging
import time
from typing import Optional, Callable
from collections import defaultdict

from backend.rag.models import DocumentChunk, ScoredChunk
from backend.rag.config import (
    MIN_BM25_SCORE_RATIO,
    MAX_CHUNKS_PER_DOC,
    MAX_CHUNKS_PER_TYPE,
    MAX_CONTEXT_TOKENS,
)
from backend.rag.utils import estimate_tokens, tokens_to_chars


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
# Gating Functions
# =============================================================================

def apply_lexical_gate(
    scored_chunks: list[ScoredChunk],
    min_ratio: Optional[float] = None,
) -> list[ScoredChunk]:
    """
    Filter out chunks with very low BM25 scores (lexical gate).
    
    Args:
        scored_chunks: List of scored chunks from retrieval
        min_ratio: Minimum BM25 score as ratio of top score
        
    Returns:
        Filtered list of scored chunks
    """
    min_ratio = min_ratio or MIN_BM25_SCORE_RATIO
    
    if not scored_chunks:
        return []
    
    # Find max BM25 score
    max_bm25 = max(sc.bm25_score for sc in scored_chunks)
    
    if max_bm25 <= 0:
        return scored_chunks  # Can't filter by BM25
    
    threshold = max_bm25 * min_ratio
    filtered = [sc for sc in scored_chunks if sc.bm25_score >= threshold]
    
    logger.debug(f"Lexical gate: {len(scored_chunks)} -> {len(filtered)} chunks (threshold={threshold:.3f})")
    return filtered


def apply_per_doc_cap(
    scored_chunks: list[ScoredChunk],
    max_per_doc: Optional[int] = None,
) -> list[ScoredChunk]:
    """
    Limit the number of chunks per document.
    
    Args:
        scored_chunks: List of scored chunks (assumed sorted by relevance)
        max_per_doc: Maximum chunks to keep per doc_id
        
    Returns:
        Filtered list respecting per-doc cap
    """
    max_per_doc = max_per_doc or MAX_CHUNKS_PER_DOC
    
    doc_counts = defaultdict(int)
    filtered = []
    
    for sc in scored_chunks:
        doc_id = sc.chunk.doc_id
        if doc_counts[doc_id] < max_per_doc:
            filtered.append(sc)
            doc_counts[doc_id] += 1
    
    logger.debug(f"Per-doc cap: {len(scored_chunks)} -> {len(filtered)} chunks")
    return filtered


def apply_per_type_cap(
    scored_chunks: list[ScoredChunk],
    max_per_type: Optional[int] = None,
) -> list[ScoredChunk]:
    """
    Limit the number of chunks per type (for private data).
    
    Args:
        scored_chunks: List of scored chunks
        max_per_type: Maximum chunks to keep per type
        
    Returns:
        Filtered list respecting per-type cap
    """
    max_per_type = max_per_type or MAX_CHUNKS_PER_TYPE
    
    # Group by type and apply cap
    by_type = defaultdict(list)
    for sc in scored_chunks:
        t = sc.chunk.metadata.get("type", "unknown")
        if len(by_type[t]) < max_per_type:
            by_type[t].append(sc)
    
    # Flatten back, maintaining original order by type priority
    result = []
    for t in ["history", "opportunity_note", "attachment", "unknown"]:
        result.extend(by_type.get(t, []))
    
    logger.debug(f"Per-type cap: {len(scored_chunks)} -> {len(result)} chunks")
    return result


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
        header = f"[{chunk.doc_id}]"
        if section:
            header += f" {section}"
        
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
