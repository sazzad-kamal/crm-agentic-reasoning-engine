"""
Text tokenizer for BM25 search with improved quality.

Provides:
- Lowercasing
- Punctuation removal
- Stop word filtering
- Basic Porter-style stemming
"""

import re
from functools import lru_cache


# Common English stop words (minimal set for search)
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "as", "if", "when",
    "than", "so", "no", "not", "only", "own", "same", "too", "very",
    "just", "also", "now", "here", "there", "where", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "any",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who", "whom",
})


# Regex for tokenization
_WORD_PATTERN = re.compile(r'\b[a-z0-9]+\b')


def _stem_word(word: str) -> str:
    """
    Apply basic Porter-style suffix stripping.

    This is a simplified stemmer that handles common English suffixes.
    For production, consider using NLTK's PorterStemmer or spaCy.
    """
    if len(word) <= 3:
        return word

    # Common suffix rules (order matters)
    suffixes = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("ization", "ize"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("ements", "e"),
        ("ments", ""),
        ("ement", "e"),
        ("ness", ""),
        ("ment", ""),
        ("able", ""),
        ("ible", ""),
        ("ance", ""),
        ("ence", ""),
        ("tion", "t"),
        ("sion", "s"),
        ("ally", "al"),
        ("izer", "ize"),
        ("ator", "ate"),
        ("ling", ""),
        ("ing", ""),
        ("ies", "y"),
        ("ied", "y"),
        ("ful", ""),
        ("ous", ""),
        ("ive", ""),
        ("ion", ""),
        ("es", ""),
        ("ed", ""),
        ("ly", ""),
        ("er", ""),
        ("s", ""),
    ]

    for suffix, replacement in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[:-len(suffix)] + replacement

    return word


@lru_cache(maxsize=10000)
def stem_word(word: str) -> str:
    """Cached word stemming."""
    return _stem_word(word)


def tokenize(text: str, use_stemming: bool = True, remove_stopwords: bool = True) -> list[str]:
    """
    Tokenize text for BM25 search.

    Args:
        text: Input text to tokenize
        use_stemming: Apply Porter-style stemming (default: True)
        remove_stopwords: Remove common stop words (default: True)

    Returns:
        List of tokens
    """
    # Lowercase and extract words
    text_lower = text.lower()
    words = _WORD_PATTERN.findall(text_lower)

    # Filter stop words
    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS]

    # Apply stemming
    if use_stemming:
        words = [stem_word(w) for w in words]

    return words


def tokenize_simple(text: str) -> list[str]:
    """
    Simple whitespace tokenizer (backward compatible).

    Use this for exact matching or when stemming is not desired.
    """
    return text.lower().split()


__all__ = [
    "tokenize",
    "tokenize_simple",
    "stem_word",
    "STOP_WORDS",
]
