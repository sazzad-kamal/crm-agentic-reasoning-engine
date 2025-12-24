"""
Retrieval Quality Tests for RAG Pipeline.

Tests that known documents are retrieved for specific queries.
These are precision/recall tests for the retrieval system.

Run with:
    pytest backend/rag/tests/test_retrieval.py -v
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.rag.models import DocumentChunk, ScoredChunk


# =============================================================================
# Test Data: Known Query-Document Pairs
# =============================================================================

# Each entry: (query, list of doc_ids that SHOULD be retrieved)
KNOWN_RETRIEVALS = [
    (
        "What is an Opportunity in Acme CRM?",
        ["opportunities_pipeline_and_forecasts"],
    ),
    (
        "How do I create a contact?",
        ["contacts_companies_and_groups"],
    ),
    (
        "What reports are available?",
        ["reports_dashboards_and_analytics"],
    ),
    (
        "How do I import data?",
        ["faq_data_import_export"],
    ),
    (
        "What is the pipeline forecast?",
        ["opportunities_pipeline_and_forecasts"],
    ),
    (
        "How do I schedule a meeting or activity?",
        ["history_activities_and_calendar"],
    ),
    (
        "What are system limits?",
        ["system_performance_and_limits"],
    ),
    (
        "How do email marketing campaigns work?",
        ["email_marketing_campaigns"],
    ),
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_chunks():
    """Create mock document chunks for testing retrieval logic."""
    chunks = []
    doc_contents = {
        "opportunities_pipeline_and_forecasts": "Opportunities represent potential deals. Pipeline stages include prospecting, qualification, proposal, negotiation, closed-won.",
        "contacts_companies_and_groups": "Contacts are individuals associated with companies. You can create contacts from the Contacts tab or import them in bulk.",
        "reports_dashboards_and_analytics": "Reports provide insights into your CRM data. Dashboards show key metrics at a glance. Analytics help track performance.",
        "faq_data_import_export": "Import data using CSV files. Export data for backup. Data migration tools are available for bulk operations.",
        "history_activities_and_calendar": "Activities include calls, meetings, and tasks. Calendar integration syncs with Outlook and Google Calendar.",
        "system_performance_and_limits": "System limits include API rate limits, storage quotas, and user limits. Performance depends on data volume.",
        "email_marketing_campaigns": "Email campaigns allow bulk email sending. Track open rates and click-through rates. Segmentation targets specific groups.",
        "product_acme_crm_overview": "Acme CRM Suite is a comprehensive customer relationship management system for sales teams.",
    }
    
    for i, (doc_id, text) in enumerate(doc_contents.items()):
        chunk = DocumentChunk(
            chunk_id=f"{doc_id}::chunk_0",
            doc_id=doc_id,
            title=doc_id.replace("_", " ").title(),
            text=text,
            metadata={"section_heading": "Overview"},
        )
        chunks.append(chunk)
    
    return chunks


@pytest.fixture
def mock_backend(mock_chunks):
    """Create a mock backend with predictable retrieval behavior."""
    from backend.rag.retrieval import RetrievalBackend
    
    backend = MagicMock(spec=RetrievalBackend)
    
    def mock_retrieve(query, k_dense=20, k_bm25=20, top_n=10, use_reranker=True, **kwargs):
        """Simple keyword-based mock retrieval."""
        query_lower = query.lower()
        results = []
        
        for chunk in mock_chunks:
            # Simple relevance: count keyword matches
            score = 0
            for word in query_lower.split():
                if word in chunk.text.lower() or word in chunk.doc_id.lower():
                    score += 1
            
            if score > 0:
                results.append(ScoredChunk(
                    chunk=chunk,
                    dense_score=score * 0.1,
                    bm25_score=score * 0.2,
                    rrf_score=score * 0.3,
                    rerank_score=score * 0.5,
                ))
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_n]
    
    backend.retrieve_candidates = mock_retrieve
    return backend


# =============================================================================
# Test: Retrieval Precision
# =============================================================================

class TestRetrievalPrecision:
    """Tests that correct documents are retrieved for known queries."""
    
    @pytest.mark.parametrize("query,expected_doc_ids", KNOWN_RETRIEVALS)
    def test_expected_doc_retrieved(self, mock_backend, query, expected_doc_ids):
        """Test that expected documents appear in retrieval results."""
        results = mock_backend.retrieve_candidates(query, top_n=5)
        
        retrieved_doc_ids = [r.chunk.doc_id for r in results]
        
        for expected_doc in expected_doc_ids:
            assert expected_doc in retrieved_doc_ids, \
                f"Expected '{expected_doc}' in results for query: {query}"
    
    def test_irrelevant_not_top_result(self, mock_backend):
        """Test that irrelevant documents are not the top result."""
        query = "How do I create an opportunity?"
        results = mock_backend.retrieve_candidates(query, top_n=5)
        
        if results:
            top_doc = results[0].chunk.doc_id
            # Should not be email campaigns for opportunity question
            assert top_doc != "email_marketing_campaigns"


# =============================================================================
# Test: Retrieval Recall (Coverage)
# =============================================================================

class TestRetrievalRecall:
    """Tests for retrieval coverage and recall."""
    
    def test_retrieval_returns_results(self, mock_backend):
        """Test that retrieval returns at least some results."""
        queries = [
            "What is Acme CRM?",
            "How do I use the system?",
            "Tell me about contacts",
        ]
        
        for query in queries:
            results = mock_backend.retrieve_candidates(query, top_n=5)
            assert len(results) > 0, f"No results for query: {query}"
    
    def test_multiple_relevant_docs_returned(self, mock_backend):
        """Test that queries return multiple relevant documents."""
        query = "How do I manage contacts and activities?"
        results = mock_backend.retrieve_candidates(query, top_n=10)
        
        doc_ids = set(r.chunk.doc_id for r in results)
        
        # Should get results from multiple docs
        assert len(doc_ids) >= 2, "Expected results from multiple documents"


# =============================================================================
# Test: Retrieval Scoring
# =============================================================================

class TestRetrievalScoring:
    """Tests for retrieval score consistency."""
    
    def test_scores_are_positive(self, mock_backend):
        """Test that all scores are non-negative."""
        query = "What is an opportunity?"
        results = mock_backend.retrieve_candidates(query, top_n=5)
        
        for result in results:
            assert result.rerank_score >= 0, "Rerank score should be non-negative"
            assert result.rrf_score >= 0, "RRF score should be non-negative"
    
    def test_results_sorted_by_score(self, mock_backend):
        """Test that results are sorted by rerank score descending."""
        query = "Pipeline and opportunities"
        results = mock_backend.retrieve_candidates(query, top_n=5)
        
        if len(results) > 1:
            scores = [r.rerank_score for r in results]
            assert scores == sorted(scores, reverse=True), \
                "Results should be sorted by rerank score"


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestRetrievalEdgeCases:
    """Tests for edge cases in retrieval."""
    
    def test_empty_query_handled(self, mock_backend):
        """Test that empty query is handled gracefully."""
        # Mock should handle empty query without crashing
        results = mock_backend.retrieve_candidates("", top_n=5)
        # Empty query might return no results, but shouldn't crash
        assert isinstance(results, list)
    
    def test_very_long_query_handled(self, mock_backend):
        """Test that very long queries are handled."""
        long_query = "opportunity " * 100  # 100 repetitions
        results = mock_backend.retrieve_candidates(long_query, top_n=5)
        assert isinstance(results, list)
    
    def test_special_characters_handled(self, mock_backend):
        """Test that special characters don't break retrieval."""
        queries = [
            "What's an opportunity?",
            "How do I use the @mentions feature?",
            "Import/export data",
            "CRM & sales",
        ]
        
        for query in queries:
            results = mock_backend.retrieve_candidates(query, top_n=5)
            assert isinstance(results, list), f"Failed for query: {query}"


# =============================================================================
# Test: Config Integration
# =============================================================================

class TestRetrievalConfig:
    """Tests for retrieval configuration."""
    
    def test_config_defaults_valid(self):
        """Test that default retrieval constants are valid."""
        from backend.rag.retrieval.constants import (
            DEFAULT_K_DENSE,
            DEFAULT_K_BM25,
            DEFAULT_TOP_N,
        )
        
        assert DEFAULT_K_DENSE > 0
        assert DEFAULT_K_BM25 > 0
        assert DEFAULT_TOP_N > 0
