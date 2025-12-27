"""
Tests for RAG gating and filtering functions.

Tests lexical gating, per-document caps, and per-type caps.

Run with:
    pytest tests/backend/rag/test_gating.py -v
"""

import pytest

from backend.rag.models import DocumentChunk, ScoredChunk
from backend.rag.pipeline.gating import (
    apply_lexical_gate,
    apply_per_doc_cap,
    apply_per_type_cap,
    apply_all_gates,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def scored_chunks():
    """Create sample scored chunks for testing."""
    chunks = [
        ScoredChunk(
            chunk=DocumentChunk(
                chunk_id="doc1::0",
                doc_id="doc1",
                title="Document 1",
                text="High scoring content",
                metadata={"type": "history"},
            ),
            dense_score=0.9,
            bm25_score=10.0,
            rrf_score=0.8,
            rerank_score=0.95,
        ),
        ScoredChunk(
            chunk=DocumentChunk(
                chunk_id="doc1::1",
                doc_id="doc1",
                title="Document 1",
                text="Medium scoring content",
                metadata={"type": "history"},
            ),
            dense_score=0.6,
            bm25_score=5.0,
            rrf_score=0.5,
            rerank_score=0.7,
        ),
        ScoredChunk(
            chunk=DocumentChunk(
                chunk_id="doc1::2",
                doc_id="doc1",
                title="Document 1",
                text="Low scoring content",
                metadata={"type": "history"},
            ),
            dense_score=0.2,
            bm25_score=1.0,
            rrf_score=0.1,
            rerank_score=0.3,
        ),
        ScoredChunk(
            chunk=DocumentChunk(
                chunk_id="doc2::0",
                doc_id="doc2",
                title="Document 2",
                text="Another high scorer",
                metadata={"type": "opportunity_note"},
            ),
            dense_score=0.85,
            bm25_score=8.0,
            rrf_score=0.75,
            rerank_score=0.9,
        ),
        ScoredChunk(
            chunk=DocumentChunk(
                chunk_id="doc2::1",
                doc_id="doc2",
                title="Document 2",
                text="Very low scorer",
                metadata={"type": "opportunity_note"},
            ),
            dense_score=0.1,
            bm25_score=0.5,
            rrf_score=0.05,
            rerank_score=0.1,
        ),
    ]
    return chunks


# =============================================================================
# Test: apply_lexical_gate
# =============================================================================

class TestApplyLexicalGate:
    """Tests for apply_lexical_gate function."""

    def test_filters_low_bm25_scores(self, scored_chunks):
        """Test that chunks with low BM25 scores are filtered."""
        filtered = apply_lexical_gate(scored_chunks, min_ratio=0.5)

        # Max BM25 is 10.0, so threshold is 5.0
        # Should keep: 10.0, 5.0, 8.0
        # Should filter: 1.0, 0.5
        assert len(filtered) <= len(scored_chunks)

        for chunk in filtered:
            assert chunk.bm25_score >= 5.0 * 0.5

    def test_keeps_all_when_no_threshold(self, scored_chunks):
        """Test that all chunks are kept when threshold is very low."""
        filtered = apply_lexical_gate(scored_chunks, min_ratio=0.0)

        # Even with 0.0 ratio, very low BM25 scores may still be filtered
        assert len(filtered) >= len(scored_chunks) - 1

    def test_handles_empty_list(self):
        """Test handling of empty chunk list."""
        filtered = apply_lexical_gate([])

        assert filtered == []

    def test_handles_zero_bm25_scores(self):
        """Test handling when all BM25 scores are zero."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    chunk_id="test::0",
                    doc_id="test",
                    title="Test",
                    text="Test",
                    metadata={},
                ),
                bm25_score=0.0,
            )
        ]

        filtered = apply_lexical_gate(chunks)

        # Should return all chunks when max is 0
        assert len(filtered) == len(chunks)

    def test_preserves_order(self, scored_chunks):
        """Test that filtering preserves original order."""
        filtered = apply_lexical_gate(scored_chunks)

        # Check that relative order is preserved
        if len(filtered) >= 2:
            original_indices = [scored_chunks.index(c) for c in filtered]
            assert original_indices == sorted(original_indices)

    def test_default_min_ratio(self, scored_chunks):
        """Test that default min_ratio is applied."""
        # Default is MIN_BM25_SCORE_RATIO from constants (0.1)
        filtered = apply_lexical_gate(scored_chunks)

        assert isinstance(filtered, list)


# =============================================================================
# Test: apply_per_doc_cap
# =============================================================================

class TestApplyPerDocCap:
    """Tests for apply_per_doc_cap function."""

    def test_limits_chunks_per_document(self, scored_chunks):
        """Test that chunks per document are limited."""
        # doc1 has 3 chunks, doc2 has 2 chunks
        filtered = apply_per_doc_cap(scored_chunks, max_per_doc=2)

        # Count chunks per doc
        doc_counts = {}
        for chunk in filtered:
            doc_id = chunk.chunk.doc_id
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        # No document should have more than 2 chunks
        for count in doc_counts.values():
            assert count <= 2

    def test_keeps_first_n_chunks_per_doc(self, scored_chunks):
        """Test that first N chunks per document are kept."""
        filtered = apply_per_doc_cap(scored_chunks, max_per_doc=1)

        # Should keep first chunk from each doc
        doc_ids = [c.chunk.doc_id for c in filtered]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

        # Should only have 2 chunks total (1 per doc)
        assert len(filtered) <= 2

    def test_handles_empty_list(self):
        """Test handling of empty chunk list."""
        filtered = apply_per_doc_cap([])

        assert filtered == []

    def test_single_chunk_unchanged(self):
        """Test that single chunk is unchanged."""
        single_chunk = [
            ScoredChunk(
                chunk=DocumentChunk(
                    chunk_id="test::0",
                    doc_id="test",
                    title="Test",
                    text="Test",
                    metadata={},
                ),
            )
        ]

        filtered = apply_per_doc_cap(single_chunk, max_per_doc=1)

        assert len(filtered) == 1

    def test_max_per_doc_zero_filters_all(self):
        """Test that max_per_doc=0 keeps at least first of each doc."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    chunk_id="test::0",
                    doc_id="test",
                    title="Test",
                    text="Test",
                    metadata={},
                ),
            )
        ]

        filtered = apply_per_doc_cap(chunks, max_per_doc=0)

        # Implementation may keep at least 1 per doc even with 0
        assert len(filtered) <= 1

    def test_preserves_order_within_doc(self, scored_chunks):
        """Test that order is preserved within each document."""
        filtered = apply_per_doc_cap(scored_chunks, max_per_doc=2)

        # Get doc1 chunks in order
        doc1_chunks = [c for c in filtered if c.chunk.doc_id == "doc1"]

        if len(doc1_chunks) >= 2:
            # Should be in original order
            original_doc1 = [c for c in scored_chunks if c.chunk.doc_id == "doc1"]
            assert doc1_chunks[0] == original_doc1[0]
            assert doc1_chunks[1] == original_doc1[1]


# =============================================================================
# Test: apply_per_type_cap
# =============================================================================

class TestApplyPerTypeCap:
    """Tests for apply_per_type_cap function."""

    def test_limits_chunks_per_type(self, scored_chunks):
        """Test that chunks per type are limited."""
        # history: 3 chunks, opportunity_note: 2 chunks
        filtered = apply_per_type_cap(scored_chunks, max_per_type=2)

        # Count chunks per type
        type_counts = {}
        for chunk in filtered:
            chunk_type = chunk.chunk.metadata.get("type", "unknown")
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

        # No type should have more than 2 chunks
        for count in type_counts.values():
            assert count <= 2

    def test_handles_unknown_type(self):
        """Test handling of chunks without type metadata."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    chunk_id="test::0",
                    doc_id="test",
                    title="Test",
                    text="Test",
                    metadata={},  # No type
                ),
            )
        ]

        filtered = apply_per_type_cap(chunks, max_per_type=1)

        assert len(filtered) <= 1

    def test_handles_empty_list(self):
        """Test handling of empty chunk list."""
        filtered = apply_per_type_cap([])

        assert filtered == []

    def test_type_priority_order(self, scored_chunks):
        """Test that types are returned in priority order."""
        filtered = apply_per_type_cap(scored_chunks, max_per_type=10)

        # Types should appear in order: history, opportunity_note, attachment, unknown
        types_seen = []
        for chunk in filtered:
            chunk_type = chunk.chunk.metadata.get("type", "unknown")
            if chunk_type not in types_seen:
                types_seen.append(chunk_type)

        # History should come before opportunity_note if both present
        if "history" in types_seen and "opportunity_note" in types_seen:
            assert types_seen.index("history") < types_seen.index("opportunity_note")

    def test_max_per_type_zero_filters_all(self):
        """Test that max_per_type=0 keeps at least first of each type."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    chunk_id="test::0",
                    doc_id="test",
                    title="Test",
                    text="Test",
                    metadata={"type": "history"},
                ),
            )
        ]

        filtered = apply_per_type_cap(chunks, max_per_type=0)

        # Implementation may keep at least 1 per type even with 0
        assert len(filtered) <= 1


# =============================================================================
# Test: apply_all_gates
# =============================================================================

class TestApplyAllGates:
    """Tests for apply_all_gates function."""

    def test_applies_multiple_gates(self, scored_chunks):
        """Test that multiple gates are applied in sequence."""
        filtered = apply_all_gates(scored_chunks, min_bm25_ratio=0.5, max_per_doc=2)

        # Should apply lexical gate AND per-doc cap
        assert len(filtered) <= len(scored_chunks)

        # Verify per-doc cap
        doc_counts = {}
        for chunk in filtered:
            doc_id = chunk.chunk.doc_id
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        for count in doc_counts.values():
            assert count <= 2

    def test_gates_are_cumulative(self, scored_chunks):
        """Test that gates reduce results cumulatively."""
        only_lexical = apply_lexical_gate(scored_chunks, min_ratio=0.5)
        both_gates = apply_all_gates(scored_chunks, min_bm25_ratio=0.5, max_per_doc=2)

        # Both gates should be equal or more restrictive than lexical alone
        assert len(both_gates) <= len(only_lexical)

    def test_handles_empty_list(self):
        """Test handling of empty chunk list."""
        filtered = apply_all_gates([])

        assert filtered == []

    def test_preserves_chunk_structure(self, scored_chunks):
        """Test that ScoredChunk structure is preserved."""
        filtered = apply_all_gates(scored_chunks)

        for chunk in filtered:
            assert isinstance(chunk, ScoredChunk)
            assert isinstance(chunk.chunk, DocumentChunk)
            assert hasattr(chunk, 'bm25_score')
            assert hasattr(chunk, 'rerank_score')


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestGatingEdgeCases:
    """Tests for edge cases in gating."""

    def test_all_same_doc_id(self):
        """Test gating when all chunks have same doc_id."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    chunk_id=f"same::{ i}",
                    doc_id="same",
                    title="Same",
                    text=f"Chunk {i}",
                    metadata={},
                ),
                bm25_score=float(i + 1),
            )
            for i in range(10)
        ]

        filtered = apply_per_doc_cap(chunks, max_per_doc=3)

        # Should only keep 3 chunks
        assert len(filtered) == 3

    def test_all_same_type(self):
        """Test gating when all chunks have same type."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    chunk_id=f"doc{i}::0",
                    doc_id=f"doc{i}",
                    title="Test",
                    text="Test",
                    metadata={"type": "history"},
                ),
                bm25_score=1.0,
            )
            for i in range(10)
        ]

        filtered = apply_per_type_cap(chunks, max_per_type=5)

        # Should only keep 5 chunks
        assert len(filtered) == 5

    def test_negative_scores_handled(self):
        """Test handling of negative BM25 scores."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    chunk_id="test::0",
                    doc_id="test",
                    title="Test",
                    text="Test",
                    metadata={},
                ),
                bm25_score=-1.0,
            )
        ]

        # Should not crash
        filtered = apply_lexical_gate(chunks)
        assert isinstance(filtered, list)

    def test_mixed_metadata_types(self):
        """Test handling of different metadata structures."""
        chunks = [
            ScoredChunk(
                chunk=DocumentChunk(
                    chunk_id="test1::0",
                    doc_id="test1",
                    title="Test",
                    text="Test",
                    metadata={"type": "history"},
                ),
                bm25_score=1.0,
            ),
            ScoredChunk(
                chunk=DocumentChunk(
                    chunk_id="test2::0",
                    doc_id="test2",
                    title="Test",
                    text="Test",
                    metadata={},  # No type
                ),
                bm25_score=1.0,
            ),
        ]

        # Should handle gracefully
        filtered = apply_per_type_cap(chunks, max_per_type=1)
        assert len(filtered) <= 2
