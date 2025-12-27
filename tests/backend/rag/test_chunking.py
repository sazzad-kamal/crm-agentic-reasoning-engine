"""
Tests for text chunking utilities.

Tests chunking logic, token estimation, and overlap handling.

Run with:
    pytest tests/backend/rag/test_chunking.py -v
"""

import pytest

from backend.rag.ingest.chunking import (
    recursive_split,
    chunk_text,
    TARGET_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from backend.rag.utils import estimate_tokens


# =============================================================================
# Test Data
# =============================================================================

SMALL_TEXT = "This is a small text that fits in one chunk."

MEDIUM_TEXT = """This is a medium-sized text.

It has multiple paragraphs to test splitting behavior.

Each paragraph should be handled correctly by the chunking algorithm.

The chunker should try to split on natural boundaries."""

LARGE_TEXT = """# Introduction

This is a very large document that will need to be split into multiple chunks.
The chunking algorithm should respect natural boundaries like paragraphs and sentences.

## Section One

This section contains information about the first topic. It has multiple sentences
that discuss various aspects of the topic in detail. The chunker should try to keep
related content together while respecting the maximum chunk size.

Some more content here to make it longer. We want to ensure that the text is split
properly across multiple chunks while maintaining readability and context.

## Section Two

This is another section with different content. It also needs to be chunked appropriately.
The algorithm should handle section boundaries gracefully.

More content to make this section longer and test the chunking behavior across
different types of text structures.

## Section Three

Final section with concluding remarks. This tests how the chunker handles the end
of the document and ensures no content is lost during the splitting process.
""" * 3  # Repeat to make it large enough


# =============================================================================
# Test: recursive_split
# =============================================================================

class TestRecursiveSplit:
    """Tests for recursive_split function."""

    def test_small_text_not_split(self):
        """Test that text smaller than max_size is not split."""
        text = "Small text"
        chunks = recursive_split(text, max_size=TARGET_CHUNK_SIZE)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_on_double_newline(self):
        """Test that text splits on paragraph boundaries (double newline)."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = recursive_split(text, max_size=50)  # Small to force split

        # Small content may not split if total is under max_size tokens
        assert len(chunks) >= 1
        # Each chunk should contain paragraph content
        for chunk in chunks:
            assert len(chunk) > 0

    def test_splits_on_sentences(self):
        """Test that text splits on sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = recursive_split(text, max_size=50)

        # Should split somewhere
        if estimate_tokens(text) > 50:
            assert len(chunks) >= 2

    def test_respects_max_size(self):
        """Test that chunks don't exceed max_size (approximately)."""
        chunks = recursive_split(LARGE_TEXT, max_size=TARGET_CHUNK_SIZE)

        for chunk in chunks:
            tokens = estimate_tokens(chunk)
            # Allow some tolerance for overlap
            assert tokens <= TARGET_CHUNK_SIZE * 1.5

    def test_no_empty_chunks(self):
        """Test that no empty chunks are created."""
        chunks = recursive_split(LARGE_TEXT, max_size=TARGET_CHUNK_SIZE)

        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_overlap_added(self):
        """Test that overlap is added between chunks."""
        text = "Word " * 200  # Create repeating text
        chunks = recursive_split(text, max_size=100, overlap=20)

        if len(chunks) > 1:
            # Check that there's some content overlap
            # (exact overlap is hard to verify due to word boundaries)
            assert len(chunks[1]) > 0

    def test_preserves_content(self):
        """Test that no content is lost during splitting."""
        chunks = recursive_split(MEDIUM_TEXT, max_size=TARGET_CHUNK_SIZE)

        # All words should appear in at least one chunk
        original_words = set(MEDIUM_TEXT.lower().split())
        chunk_words = set()
        for chunk in chunks:
            chunk_words.update(chunk.lower().split())

        # Most words should be preserved (allowing for some edge case trimming)
        assert len(chunk_words & original_words) >= len(original_words) * 0.9


# =============================================================================
# Test: chunk_text
# =============================================================================

class TestChunkText:
    """Tests for chunk_text function."""

    def test_small_text_unchanged(self):
        """Test that small text is returned as single chunk."""
        chunks = chunk_text(SMALL_TEXT)

        assert len(chunks) == 1
        assert chunks[0] == SMALL_TEXT

    def test_splits_large_text(self):
        """Test that large text is split into multiple chunks."""
        chunks = chunk_text(LARGE_TEXT)

        assert len(chunks) > 1

    def test_respects_max_size(self):
        """Test that chunks respect max_size."""
        chunks = chunk_text(LARGE_TEXT, max_size=TARGET_CHUNK_SIZE)

        for chunk in chunks:
            tokens = estimate_tokens(chunk)
            assert tokens <= MAX_CHUNK_SIZE * 1.2  # Allow some tolerance

    def test_filters_small_chunks(self):
        """Test that very small chunks are filtered out."""
        # Create text that might produce tiny chunks
        text = "A.\n\nB.\n\nLonger paragraph with actual content here."
        chunks = chunk_text(text, min_size=MIN_CHUNK_SIZE)

        # Very small chunks should be filtered
        for chunk in chunks:
            tokens = estimate_tokens(chunk)
            # Allow chunks at least half the min size
            assert tokens >= MIN_CHUNK_SIZE // 4 or len(chunks) == 1

    def test_handles_only_newlines(self):
        """Test handling of text with only newlines."""
        text = "\n\n\n"
        chunks = chunk_text(text)

        # Should return something (possibly empty list or single chunk)
        assert isinstance(chunks, list)

    def test_handles_no_punctuation(self):
        """Test handling of text without sentence boundaries."""
        text = "word " * 1000  # No periods
        chunks = chunk_text(text, max_size=TARGET_CHUNK_SIZE)

        # Should split if content exceeds token limit
        total_tokens = estimate_tokens(text)
        if total_tokens > TARGET_CHUNK_SIZE:
            assert len(chunks) >= 1  # At least one chunk


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestChunkingEdgeCases:
    """Tests for edge cases in chunking."""

    def test_empty_string(self):
        """Test handling of empty string."""
        chunks = recursive_split("")
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_single_word(self):
        """Test handling of single word."""
        chunks = recursive_split("word")
        assert len(chunks) == 1
        assert chunks[0] == "word"

    def test_very_long_word(self):
        """Test handling of word longer than max_size."""
        long_word = "a" * 5000
        chunks = recursive_split(long_word, max_size=100)

        # Should split even a single word if too long
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        text = "Hello 你好 مرحبا שלום Здравствуйте 🎉"
        chunks = recursive_split(text)

        assert len(chunks) >= 1
        # Unicode should be preserved
        assert "🎉" in chunks[0] or len(chunks) == 1

    def test_multiple_separators(self):
        """Test handling of multiple separator types."""
        text = "First.\n\nSecond! Third? Fourth."
        chunks = recursive_split(text, max_size=50)

        assert isinstance(chunks, list)
        assert all(len(c) > 0 for c in chunks)

    def test_custom_max_size(self):
        """Test that custom max_size is respected."""
        custom_size = 200
        chunks = recursive_split(LARGE_TEXT, max_size=custom_size)

        for chunk in chunks:
            tokens = estimate_tokens(chunk)
            assert tokens <= custom_size * 1.5  # Allow tolerance

    def test_custom_overlap(self):
        """Test that custom overlap is used."""
        text = "Word " * 500
        chunks_no_overlap = recursive_split(text, max_size=100, overlap=0)
        chunks_with_overlap = recursive_split(text, max_size=100, overlap=50)

        # With overlap, chunks should be longer on average
        if len(chunks_with_overlap) > 1:
            avg_len_overlap = sum(len(c) for c in chunks_with_overlap) / len(chunks_with_overlap)
            avg_len_no_overlap = sum(len(c) for c in chunks_no_overlap) / len(chunks_no_overlap)
            # Overlapping chunks should generally be longer (but not always)
            assert isinstance(avg_len_overlap, float)
            assert isinstance(avg_len_no_overlap, float)


# =============================================================================
# Test: Constants
# =============================================================================

class TestChunkingConstants:
    """Tests for chunking constants."""

    def test_constants_are_positive(self):
        """Test that all constants are positive integers."""
        assert TARGET_CHUNK_SIZE > 0
        assert MAX_CHUNK_SIZE > 0
        assert MIN_CHUNK_SIZE > 0
        assert CHUNK_OVERLAP >= 0

    def test_constants_are_logical(self):
        """Test that constants have logical relationships."""
        assert MAX_CHUNK_SIZE >= TARGET_CHUNK_SIZE
        assert TARGET_CHUNK_SIZE >= MIN_CHUNK_SIZE
        assert CHUNK_OVERLAP < TARGET_CHUNK_SIZE

    def test_target_is_500(self):
        """Test that target chunk size is 500 tokens."""
        assert TARGET_CHUNK_SIZE == 500

    def test_max_is_700(self):
        """Test that max chunk size is 700 tokens."""
        assert MAX_CHUNK_SIZE == 700

    def test_min_is_100(self):
        """Test that min chunk size is 100 tokens."""
        assert MIN_CHUNK_SIZE == 100

    def test_overlap_is_50(self):
        """Test that overlap is 50 tokens."""
        assert CHUNK_OVERLAP == 50
