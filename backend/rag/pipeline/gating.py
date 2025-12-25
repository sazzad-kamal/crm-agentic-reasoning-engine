"""
Gating and filtering functions for RAG pipeline.

This module provides functions to filter retrieved chunks based on
various criteria like lexical scores, per-document caps, and per-type caps.
"""

import logging
from collections import defaultdict
from typing import Optional

from ..models import ScoredChunk
from .constants import MIN_BM25_SCORE_RATIO, MAX_CHUNKS_PER_DOC, MAX_CHUNKS_PER_TYPE

logger = logging.getLogger(__name__)


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
    
    doc_counts: dict[str, int] = defaultdict(int)
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
    by_type: dict[str, list[ScoredChunk]] = defaultdict(list)
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


def apply_all_gates(
    scored_chunks: list[ScoredChunk],
    min_bm25_ratio: Optional[float] = None,
    max_per_doc: Optional[int] = None,
) -> list[ScoredChunk]:
    """
    Apply lexical gate and per-doc cap in sequence.
    
    Args:
        scored_chunks: List of scored chunks
        min_bm25_ratio: Minimum BM25 score ratio for lexical gate
        max_per_doc: Maximum chunks per document
        
    Returns:
        Filtered list of scored chunks
    """
    result = apply_lexical_gate(scored_chunks, min_bm25_ratio)
    result = apply_per_doc_cap(result, max_per_doc)
    return result
