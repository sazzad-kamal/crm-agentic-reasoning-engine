"""
Shared utilities for the RAG pipeline.

Contains common functions used across multiple modules to avoid duplication:
- Token estimation
- Text chunking (recursive splitting)
- Query preprocessing
- Text normalization
"""

import re
import logging
from typing import Optional

from backend.rag.config import get_config


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text length.
    
    Uses a simple character-based heuristic. For more accurate counts,
    use tiktoken with the specific model's tokenizer.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated number of tokens
    """
    config = get_config()
    return len(text) // config.chars_per_token


def tokens_to_chars(tokens: int) -> int:
    """Convert token count to approximate character count."""
    config = get_config()
    return tokens * config.chars_per_token


# =============================================================================
# Query Preprocessing
# =============================================================================

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


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.
    
    - Strip whitespace
    - Normalize unicode
    - Collapse multiple whitespace
    
    Args:
        text: Raw text
        
    Returns:
        Normalized text
    """
    import unicodedata
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Strip and collapse whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text


# =============================================================================
# Text Chunking
# =============================================================================

def recursive_split(
    text: str,
    max_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[str]:
    """
    Recursively split text into chunks of approximately max_size tokens.
    
    Tries to split on natural boundaries in order:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences (. ! ?)
    4. Words (spaces)
    
    Args:
        text: Text to split
        max_size: Maximum chunk size in tokens (default from config)
        overlap: Overlap between chunks in tokens (default from config)
        
    Returns:
        List of text chunks
    """
    config = get_config()
    max_size = max_size or config.target_chunk_size
    overlap = overlap or config.chunk_overlap
    
    estimated_tokens = estimate_tokens(text)
    
    if estimated_tokens <= max_size:
        return [text]
    
    # Try splitting by different separators
    separators = ["\n\n", "\n", ". ", "! ", "? ", " "]
    
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current_chunk = ""
            
            for part in parts:
                test_chunk = current_chunk + sep + part if current_chunk else part
                if estimate_tokens(test_chunk) <= max_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk)
            
            if len(chunks) > 1:
                # Add overlap between chunks
                result = []
                for i, chunk in enumerate(chunks):
                    if i > 0 and overlap > 0:
                        # Add some content from the end of previous chunk
                        prev_words = chunks[i-1].split()[-overlap:]
                        chunk = " ".join(prev_words) + " " + chunk
                    result.append(chunk.strip())
                return result
    
    # If nothing worked, just split by character count
    char_limit = tokens_to_chars(max_size)
    overlap_chars = tokens_to_chars(overlap)
    chunks = []
    for i in range(0, len(text), char_limit - overlap_chars):
        chunks.append(text[i:i + char_limit])
    
    return chunks


def chunk_text(
    text: str,
    max_size: Optional[int] = None,
    min_size: Optional[int] = None,
) -> list[str]:
    """
    Split text into chunks, optimized for shorter CRM texts.
    
    Similar to recursive_split but with handling for very short texts
    that don't need splitting.
    
    Args:
        text: Text to chunk
        max_size: Maximum chunk size in tokens
        min_size: Minimum chunk size in tokens
        
    Returns:
        List of text chunks
    """
    config = get_config()
    max_size = max_size or config.target_chunk_size
    min_size = min_size or config.min_chunk_size
    
    estimated = estimate_tokens(text)
    
    # If text is small enough, return as single chunk
    if estimated <= max_size:
        return [text]
    
    # Try splitting by paragraphs first
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    
    for para in paragraphs:
        test = current + "\n\n" + para if current else para
        if estimate_tokens(test) <= max_size:
            current = test
        else:
            if current:
                chunks.append(current.strip())
            current = para
    
    if current:
        chunks.append(current.strip())
    
    # If still too large, split by sentences/lines
    result = []
    for chunk in chunks:
        if estimate_tokens(chunk) > max_size:
            # Split by newlines or sentences
            lines = chunk.replace(". ", ".\n").split("\n")
            sub_chunk = ""
            for line in lines:
                test = sub_chunk + " " + line if sub_chunk else line
                if estimate_tokens(test) <= max_size:
                    sub_chunk = test
                else:
                    if sub_chunk:
                        result.append(sub_chunk.strip())
                    sub_chunk = line
            if sub_chunk:
                result.append(sub_chunk.strip())
        else:
            result.append(chunk)
    
    # Filter out chunks that are too small
    result = [c for c in result if estimate_tokens(c) >= min_size // 2]
    
    return result if result else [text]


# =============================================================================
# Citation Extraction
# =============================================================================

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


# =============================================================================
# Text Similarity Helpers
# =============================================================================

def simple_tokenize(text: str) -> list[str]:
    """
    Simple whitespace tokenizer for BM25.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of lowercase tokens
    """
    return text.lower().split()


def compute_overlap(text1: str, text2: str) -> float:
    """
    Compute word overlap ratio between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    words1 = set(simple_tokenize(text1))
    words2 = set(simple_tokenize(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0
