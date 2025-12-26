"""
Text chunking utilities for document ingestion.
"""

from backend.rag.utils import CHARS_PER_TOKEN, estimate_tokens, tokens_to_chars
from backend.rag.ingest.constants import (
    TARGET_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def _tokens_to_chars(tokens: int) -> int:
    """Convert token count to approximate character count."""
    return tokens * CHARS_PER_TOKEN


def recursive_split(
    text: str,
    max_size: int | None = None,
    overlap: int | None = None,
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
    max_size = max_size or TARGET_CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP
    
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
    char_limit = _tokens_to_chars(max_size)
    overlap_chars = _tokens_to_chars(overlap)
    chunks = []
    for i in range(0, len(text), char_limit - overlap_chars):
        chunks.append(text[i:i + char_limit])
    
    return chunks


def chunk_text(
    text: str,
    max_size: int | None = None,
    min_size: int | None = None,
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
    max_size = max_size or TARGET_CHUNK_SIZE
    min_size = min_size or MIN_CHUNK_SIZE
    
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
