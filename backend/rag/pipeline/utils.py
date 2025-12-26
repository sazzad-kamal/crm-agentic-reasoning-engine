"""
Pipeline utilities for RAG answer generation.
"""

import re

from backend.rag.utils import CHARS_PER_TOKEN, estimate_tokens, tokens_to_chars


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
    # Remove duplicates preserving order (case-insensitive)
    return list({c.lower(): c for c in citations}.values())
