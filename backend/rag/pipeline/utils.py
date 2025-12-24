"""
Pipeline utilities for RAG answer generation.
"""

import re

from backend.rag.ingest.chunking import estimate_tokens
from backend.rag.ingest.constants import CHARS_PER_TOKEN

# Re-export for backward compatibility
__all__ = ["estimate_tokens", "tokens_to_chars", "preprocess_query", "extract_citations"]


def tokens_to_chars(tokens: int) -> int:
    """Convert token count to approximate character count."""
    return tokens * CHARS_PER_TOKEN


def preprocess_query(query: str) -> str:
    """
    Light preprocessing of user queries.
    
    - Strip whitespace
    - Collapse multiple spaces
    - Remove excessive punctuation
    
    Args:
        query: Raw user query
        
    Returns:
        Cleaned query string
    """
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    return query


def extract_citations(text: str) -> list[str]:
    """
    Extract citation references from generated text.
    
    Looks for [doc_id] or [source_id] patterns.
    
    Args:
        text: Text potentially containing citations
        
    Returns:
        List of unique cited document/source IDs
    """
    pattern = r'\[([a-zA-Z0-9_::\-]+)\]'
    citations = re.findall(pattern, text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for c in citations:
        c_lower = c.lower()
        if c_lower not in seen:
            seen.add(c_lower)
            unique.append(c)
    
    return unique
