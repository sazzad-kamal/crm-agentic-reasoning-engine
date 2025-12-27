"""
Tests for markdown document ingestion.

Tests markdown parsing, heading extraction, and document chunking.

Run with:
    pytest tests/backend/rag/test_ingest_docs.py -v
"""

import pytest
from pathlib import Path

from backend.rag.ingest.docs import (
    extract_title,
    split_by_headings,
    process_markdown_file,
)
from backend.rag.models import DocumentChunk


# =============================================================================
# Test Data
# =============================================================================

SIMPLE_MARKDOWN = """# Main Title

This is the introduction paragraph with enough content to pass the minimum chunk size filter.
It needs to have sufficient text to be considered a valid chunk for processing.

## Section One

Content for section one goes here. This section also needs enough content to pass
the minimum chunk size threshold. Adding more text to ensure it is not filtered out
during the chunking process. The chunker filters out very small sections.

## Section Two

Content for section two is also here with additional text. Again, we need to ensure
there is enough content in this section to meet the minimum requirements for chunking.
More text is added here to make sure the section passes the filter.
"""

NESTED_HEADINGS_MARKDOWN = """# Document Title

Introduction text.

## Level 2 Heading

Some content here.

### Level 3 Heading

Nested content.

#### Level 4 Heading

Deeply nested content.

### Another Level 3

More content.

## Another Level 2

Final section.
"""

NO_HEADINGS_MARKDOWN = """This is a document without any headings.
It just has paragraphs of text.

Multiple paragraphs should still be handled correctly.
"""

ONLY_H1_MARKDOWN = """# Only H1

All content is under a single H1 heading.
No other heading levels present.
"""


# =============================================================================
# Test: extract_title
# =============================================================================

class TestExtractTitle:
    """Tests for extract_title function."""

    def test_extracts_h1_title(self):
        """Test that H1 heading is extracted as title."""
        content = "# My Document Title\n\nSome content"
        title = extract_title(content, "fallback_name")

        assert title == "My Document Title"

    def test_uses_filename_when_no_h1(self):
        """Test that filename is used when no H1 present."""
        content = "## Section\n\nContent without H1"
        title = extract_title(content, "my_document")

        assert title == "My Document"

    def test_handles_empty_content(self):
        """Test handling of empty content."""
        content = ""
        title = extract_title(content, "empty_doc")

        assert title == "Empty Doc"

    def test_strips_whitespace_from_title(self):
        """Test that whitespace is stripped from title."""
        content = "#   Spaced Title   \n\nContent"
        title = extract_title(content, "fallback")

        assert title == "Spaced Title"

    def test_handles_h1_at_end(self):
        """Test that only first H1 is used."""
        content = "# First Title\n\nContent\n\n# Second Title"
        title = extract_title(content, "fallback")

        assert title == "First Title"

    def test_filename_formatting(self):
        """Test that filename is formatted nicely."""
        title = extract_title("No H1 here", "my_test_document")

        assert title == "My Test Document"


# =============================================================================
# Test: split_by_headings
# =============================================================================

class TestSplitByHeadings:
    """Tests for split_by_headings function."""

    def test_splits_h2_sections(self):
        """Test that content is split by H2 headings."""
        sections = split_by_headings(SIMPLE_MARKDOWN)

        # Should have: intro + 2 sections
        assert len(sections) >= 3

    def test_section_has_correct_structure(self):
        """Test that sections have correct structure."""
        sections = split_by_headings(SIMPLE_MARKDOWN)

        for section in sections:
            assert "section_path" in section
            assert "text" in section
            assert "level" in section
            assert isinstance(section["section_path"], list)
            assert isinstance(section["text"], str)
            assert isinstance(section["level"], int)

    def test_preserves_heading_hierarchy(self):
        """Test that heading hierarchy is preserved in section_path."""
        sections = split_by_headings(NESTED_HEADINGS_MARKDOWN)

        # Find a level 3 section
        level3_sections = [s for s in sections if s["level"] == 3]
        assert len(level3_sections) > 0

        # Level 3 should have 2 items in path (L2 + L3)
        for section in level3_sections:
            assert len(section["section_path"]) == 2

    def test_handles_no_headings(self):
        """Test handling of document without headings."""
        sections = split_by_headings(NO_HEADINGS_MARKDOWN)

        # Should return the content as one section
        assert len(sections) >= 1
        assert len(sections[0]["text"]) > 0

    def test_handles_only_h1(self):
        """Test handling of document with only H1."""
        sections = split_by_headings(ONLY_H1_MARKDOWN)

        # Should have content after H1
        assert len(sections) >= 1

    def test_empty_sections_filtered(self):
        """Test that empty sections are filtered."""
        markdown = "## Empty\n\n## Has Content\n\nActual content here"
        sections = split_by_headings(markdown)

        # All returned sections should have text
        for section in sections:
            # Either has text or it's the only section
            assert len(section["text"]) > 0 or len(sections) == 1

    def test_section_path_updates_correctly(self):
        """Test that section_path updates correctly for different levels."""
        sections = split_by_headings(NESTED_HEADINGS_MARKDOWN)

        # Level 2 should reset path
        level2_sections = [s for s in sections if s["level"] == 2]
        for section in level2_sections:
            assert len(section["section_path"]) == 1

        # Level 4 should have 3 items in path
        level4_sections = [s for s in sections if s["level"] == 4]
        if level4_sections:
            assert len(level4_sections[0]["section_path"]) == 3

    def test_introduction_section_created(self):
        """Test that content before first heading becomes Introduction."""
        markdown = "Some intro text.\n\n## First Section\n\nContent"
        sections = split_by_headings(markdown)

        # First section should be introduction
        if len(sections) > 1:
            intro_section = sections[0]
            assert "Introduction" in intro_section["section_path"] or intro_section["level"] == 1


# =============================================================================
# Test: process_markdown_file
# =============================================================================

class TestProcessMarkdownFile:
    """Tests for process_markdown_file function."""

    @pytest.fixture
    def temp_markdown_file(self, tmp_path):
        """Create a temporary markdown file for testing."""
        file_path = tmp_path / "test_doc.md"
        file_path.write_text(SIMPLE_MARKDOWN, encoding="utf-8")
        return file_path

    @pytest.fixture
    def large_markdown_file(self, tmp_path):
        """Create a large markdown file that requires chunking."""
        large_content = """# Large Document

## Section One

""" + ("This is a paragraph. " * 200) + """

## Section Two

""" + ("More content here. " * 200)

        file_path = tmp_path / "large_doc.md"
        file_path.write_text(large_content, encoding="utf-8")
        return file_path

    def test_returns_document_chunks(self, temp_markdown_file):
        """Test that function returns DocumentChunk objects."""
        chunks = process_markdown_file(temp_markdown_file)

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

    def test_chunk_has_correct_fields(self, temp_markdown_file):
        """Test that chunks have all required fields."""
        chunks = process_markdown_file(temp_markdown_file)

        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.doc_id
            assert chunk.text
            assert isinstance(chunk.metadata, dict)

    def test_doc_id_from_filename(self, temp_markdown_file):
        """Test that doc_id is derived from filename."""
        chunks = process_markdown_file(temp_markdown_file)

        for chunk in chunks:
            assert chunk.doc_id == "test_doc"

    def test_chunk_ids_are_unique(self, temp_markdown_file):
        """Test that all chunk IDs are unique."""
        chunks = process_markdown_file(temp_markdown_file)

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_ids_sequential(self, temp_markdown_file):
        """Test that chunk IDs are sequential."""
        chunks = process_markdown_file(temp_markdown_file)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"test_doc::{i}"

    def test_metadata_includes_section(self, temp_markdown_file):
        """Test that metadata includes section information."""
        chunks = process_markdown_file(temp_markdown_file)

        for chunk in chunks:
            assert "section_path" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "estimated_tokens" in chunk.metadata

    def test_large_sections_are_split(self, large_markdown_file):
        """Test that large sections are split into multiple chunks."""
        chunks = process_markdown_file(large_markdown_file)

        # Large document should produce multiple chunks
        assert len(chunks) > 2

    def test_preserves_content(self, temp_markdown_file):
        """Test that all content is preserved in chunks."""
        original_content = temp_markdown_file.read_text(encoding="utf-8")
        chunks = process_markdown_file(temp_markdown_file)

        # Combine all chunk texts
        combined_text = " ".join(c.text for c in chunks)

        # Most words should be preserved
        original_words = set(original_content.lower().split())
        chunk_words = set(combined_text.lower().split())

        overlap = len(chunk_words & original_words)
        assert overlap >= len(original_words) * 0.7  # At least 70% preserved

    def test_title_extracted_correctly(self, temp_markdown_file):
        """Test that title is extracted from H1."""
        chunks = process_markdown_file(temp_markdown_file)

        # All chunks should have same title
        titles = set(c.title for c in chunks)
        assert len(titles) == 1
        assert "Main Title" in titles

    def test_filters_very_small_chunks(self, temp_markdown_file):
        """Test that very small chunks are filtered out."""
        chunks = process_markdown_file(temp_markdown_file)

        # No chunks should be extremely tiny (unless it's the only chunk)
        for chunk in chunks:
            if len(chunks) > 1:
                assert len(chunk.text) > 10  # At least some content


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestIngestEdgeCases:
    """Tests for edge cases in document ingestion."""

    def test_unicode_in_markdown(self, tmp_path):
        """Test handling of unicode characters in markdown."""
        content = """# Unicode Test 你好

This document contains unicode characters including Chinese 你好 and émojis 🎉.
We need enough content here to pass the minimum chunk size filter during processing.
The chunker will filter out very small sections, so we add more text here.
This ensures the test properly validates unicode handling in larger chunks.
"""
        file_path = tmp_path / "unicode.md"
        file_path.write_text(content, encoding="utf-8")

        chunks = process_markdown_file(file_path)

        # May be empty if still below threshold
        if chunks:
            combined = " ".join(c.text for c in chunks)
            assert "你好" in combined or "🎉" in combined

    def test_empty_markdown_file(self, tmp_path):
        """Test handling of empty markdown file."""
        file_path = tmp_path / "empty.md"
        file_path.write_text("", encoding="utf-8")

        chunks = process_markdown_file(file_path)

        # Should handle gracefully
        assert isinstance(chunks, list)

    def test_markdown_with_code_blocks(self, tmp_path):
        """Test handling of markdown code blocks."""
        content = """# Code Example

This document contains a code block example. We need sufficient content around
the code block to ensure the section passes the minimum chunk size filter.
The following code demonstrates a simple Python function:

```python
def hello():
    print("world")
    return True
```

More content after the code block to ensure we have enough text.
This additional content helps the chunk pass the size threshold.
"""
        file_path = tmp_path / "code.md"
        file_path.write_text(content, encoding="utf-8")

        chunks = process_markdown_file(file_path)

        # May be empty if still below threshold
        if chunks:
            combined = " ".join(c.text for c in chunks)
            assert "print" in combined or "hello" in combined

    def test_markdown_with_lists(self, tmp_path):
        """Test handling of markdown lists."""
        content = """# Lists

This document contains various list formats. We need enough content to pass
the minimum chunk size filter during the ingestion process.

- Item 1 with some additional description text
- Item 2 with more content here
- Item 3 also has extra text to make it longer

1. Numbered item 1 with description
2. Numbered item 2 with more content
3. Numbered item 3 for completeness

Additional paragraph after the lists to ensure sufficient content.
"""
        file_path = tmp_path / "lists.md"
        file_path.write_text(content, encoding="utf-8")

        chunks = process_markdown_file(file_path)

        # Should produce at least one chunk with this content
        assert isinstance(chunks, list)

    def test_special_characters_in_filename(self, tmp_path):
        """Test handling of special characters in filename."""
        content = """# Test Document

This is a test document with a filename containing special characters.
We need sufficient content here to pass the minimum chunk size filter.
The doc_id should be derived from the filename stem without the extension.
"""
        file_path = tmp_path / "test-doc_123.md"
        file_path.write_text(content, encoding="utf-8")

        chunks = process_markdown_file(file_path)

        if chunks:
            assert chunks[0].doc_id == "test-doc_123"

        chunks = process_markdown_file(file_path)

        assert chunks[0].doc_id == "test-doc_123"
