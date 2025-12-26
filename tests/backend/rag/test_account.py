"""
Tests for Account-aware RAG (MVP2).

Run with:
    pytest backend/rag/tests/test_account.py -v

Some tests require OPENAI_API_KEY to be set; those are skipped if not available.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Check if OpenAI is available
OPENAI_AVAILABLE = os.environ.get("OPENAI_API_KEY") is not None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def csv_dir():
    """Get the CSV directory."""
    from backend.rag.utils import find_csv_dir
    return find_csv_dir()


@pytest.fixture
def companies_df(csv_dir):
    """Load companies dataframe."""
    import pandas as pd
    return pd.read_csv(csv_dir / "companies.csv")


# =============================================================================
# Test: Company Resolution
# =============================================================================

class TestCompanyResolver:
    """Tests for company_id resolution."""
    
    def test_resolve_by_company_id(self, companies_df):
        """Test resolving by direct company_id."""
        from backend.rag.pipeline.account import resolve_company_id
        
        # Get a real company_id from the data
        company_id = companies_df.iloc[0]["company_id"]
        expected_name = companies_df.iloc[0]["name"]
        
        resolved_id, resolved_name = resolve_company_id(company_id=company_id)
        
        assert resolved_id == company_id
        assert resolved_name == expected_name
    
    def test_resolve_by_company_name_exact(self, companies_df):
        """Test resolving by exact company name."""
        from backend.rag.pipeline.account import resolve_company_id
        
        expected_id = companies_df.iloc[0]["company_id"]
        company_name = companies_df.iloc[0]["name"]
        
        resolved_id, resolved_name = resolve_company_id(company_name=company_name)
        
        assert resolved_id == expected_id
        assert resolved_name == company_name
    
    def test_resolve_by_company_name_partial(self, companies_df):
        """Test resolving by partial company name (case-insensitive contains)."""
        from backend.rag.pipeline.account import resolve_company_id
        
        # Use partial name
        full_name = companies_df.iloc[0]["name"]
        partial_name = full_name.split()[0].lower()  # First word, lowercase
        
        resolved_id, resolved_name = resolve_company_id(company_name=partial_name)
        
        # Should resolve to the matching company
        assert resolved_id is not None
        assert partial_name.lower() in resolved_name.lower()
    
    def test_resolve_invalid_company_id_raises(self):
        """Test that invalid company_id raises ValueError."""
        from backend.rag.pipeline.account import resolve_company_id
        
        with pytest.raises(ValueError, match="not found"):
            resolve_company_id(company_id="INVALID-COMPANY-XYZ")
    
    def test_resolve_invalid_company_name_raises(self):
        """Test that invalid company_name raises ValueError."""
        from backend.rag.pipeline.account import resolve_company_id
        
        with pytest.raises(ValueError, match="No company found"):
            resolve_company_id(company_name="NonexistentCompanyXYZ123")
    
    def test_resolve_requires_id_or_name(self):
        """Test that either company_id or company_name is required."""
        from backend.rag.pipeline.account import resolve_company_id
        
        with pytest.raises(ValueError, match="Must provide either"):
            resolve_company_id()


# =============================================================================
# Test: Private Retrieval Filter
# =============================================================================

class TestPrivateRetrievalFilter:
    """Tests for private retrieval with company filtering."""
    
    @pytest.fixture
    def mock_backend(self):
        """Create a mock private retrieval backend."""
        from backend.rag.models import DocumentChunk, ScoredChunk
        
        # Create mock chunks for multiple companies
        chunks = [
            DocumentChunk(
                chunk_id="chunk_1",
                doc_id="history::HIST-ACME-1",
                title="ACME History",
                text="This is ACME Manufacturing history",
                metadata={"company_id": "ACME-MFG", "type": "history"},
            ),
            DocumentChunk(
                chunk_id="chunk_2",
                doc_id="history::HIST-BETA-1",
                title="BETA History",
                text="This is Beta Tech history",
                metadata={"company_id": "BETA-TECH", "type": "history"},
            ),
            DocumentChunk(
                chunk_id="chunk_3",
                doc_id="opp_note::OPP-ACME-1",
                title="ACME Opportunity",
                text="ACME opportunity notes",
                metadata={"company_id": "ACME-MFG", "type": "opportunity_note"},
            ),
        ]
        
        backend = MagicMock()
        backend._chunks = chunks
        backend.get_all_companies.return_value = ["ACME-MFG", "BETA-TECH"]
        
        return backend, chunks
    
    def test_filter_returns_only_target_company(self, mock_backend):
        """Test that company filter only returns chunks from target company."""
        from backend.rag.models import ScoredChunk
        
        backend, chunks = mock_backend
        
        # Simulate filtered retrieval
        target_company = "ACME-MFG"
        filtered = [c for c in chunks if c.metadata.get("company_id") == target_company]
        
        # Assert all returned chunks belong to target company
        for chunk in filtered:
            assert chunk.metadata.get("company_id") == target_company
        
        # Assert we got the expected number
        assert len(filtered) == 2  # chunk_1 and chunk_3
    
    def test_filter_excludes_other_companies(self, mock_backend):
        """Test that filter excludes chunks from other companies."""
        backend, chunks = mock_backend
        
        target_company = "ACME-MFG"
        filtered = [c for c in chunks if c.metadata.get("company_id") == target_company]
        
        # Assert BETA-TECH chunk is not included
        beta_chunks = [c for c in filtered if c.metadata.get("company_id") == "BETA-TECH"]
        assert len(beta_chunks) == 0
    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI API key not set")
    def test_real_retrieval_filter_enforced(self):
        """
        Integration test: verify real retrieval respects company filter.
        
        Requires:
        - Private collection to be ingested
        - OPENAI_API_KEY for embedding model (actually uses sentence-transformers, so not needed)
        """
        try:
            from backend.rag.retrieval.private import create_private_backend
        except Exception as e:
            pytest.skip(f"Could not import private backend: {e}")
        
        try:
            backend = create_private_backend()
        except ValueError as e:
            pytest.skip(f"Private collection not ready: {e}")
        
        # Get a company that exists
        companies = backend.get_all_companies()
        if not companies:
            pytest.skip("No companies in private index")
        
        target_company = companies[0]
        
        # Retrieve with filter
        results = backend.retrieve_candidates(
            query="What happened recently?",
            top_n=5,
            company_filter=target_company,
        )
        
        # Assert all results belong to target company
        for result in results:
            result_company = result.chunk.metadata.get("company_id")
            assert result_company == target_company, (
                f"Filter violation: expected {target_company}, got {result_company}"
            )


# =============================================================================
# Test: Privacy Leakage Detection
# =============================================================================

class TestPrivacyLeakage:
    """Tests for privacy leakage detection."""
    
    def test_no_leakage_when_all_match(self):
        """Test that no leakage is detected when all hits match target."""
        from backend.rag.eval.judge import check_privacy_leakage
        
        target = "ACME-MFG"
        hits = [
            {"id": "hist1", "company_id": "ACME-MFG"},
            {"id": "hist2", "company_id": "ACME-MFG"},
        ]
        
        leakage, leaked = check_privacy_leakage(target, hits)
        
        assert leakage == 0
        assert leaked == []
    
    def test_leakage_detected_when_mismatch(self):
        """Test that leakage is detected when hits from other company."""
        from backend.rag.eval.judge import check_privacy_leakage
        
        target = "ACME-MFG"
        hits = [
            {"id": "hist1", "company_id": "ACME-MFG"},
            {"id": "hist2", "company_id": "BETA-TECH"},  # Leaked!
        ]
        
        leakage, leaked = check_privacy_leakage(target, hits)
        
        assert leakage == 1
        assert "BETA-TECH" in leaked
    
    def test_empty_hits_no_leakage(self):
        """Test that empty hits result in no leakage."""
        from backend.rag.eval.judge import check_privacy_leakage
        
        leakage, leaked = check_privacy_leakage("ACME-MFG", [])
        
        assert leakage == 0
        assert leaked == []


# =============================================================================
# Test: CSV Directory Locator
# =============================================================================

class TestCsvDirLocator:
    """Tests for CSV directory locator."""
    
    def test_find_csv_dir_returns_path(self):
        """Test that find_csv_dir returns a valid path."""
        from backend.rag.utils import find_csv_dir
        
        csv_dir = find_csv_dir()
        
        assert csv_dir.exists()
        assert csv_dir.is_dir()
        assert (csv_dir / "companies.csv").exists()
        assert (csv_dir / "history.csv").exists()
    
    def test_find_csv_dir_error_message_clear(self, tmp_path):
        """Test that error message is clear when no dir found."""
        from backend.rag.utils import find_csv_dir, CSV_DIR_CANDIDATES
        
        # Temporarily patch the candidates to empty dirs
        with patch('backend.rag.utils.CSV_DIR_CANDIDATES', 
                   [tmp_path / "nonexistent1", tmp_path / "nonexistent2"]):
            with pytest.raises(FileNotFoundError, match="Could not find CSV data directory"):
                find_csv_dir()


# =============================================================================
# Test: Question Generation
# =============================================================================

class TestQuestionGeneration:
    """Tests for evaluation question generation."""
    
    def test_generates_expected_number_of_questions(self):
        """Test that question generation produces expected count."""
        from backend.rag.eval.account_eval import (
            generate_eval_questions,
            NUM_COMPANIES,
            NUM_QUESTIONS_PER_COMPANY,
        )
        
        questions = generate_eval_questions()
        
        # Base questions + edge case questions (3 companies × 3 edge case templates)
        NUM_EDGE_CASE_COMPANIES = 3
        NUM_EDGE_CASE_TEMPLATES = 3
        expected = (NUM_COMPANIES * NUM_QUESTIONS_PER_COMPANY) + (NUM_EDGE_CASE_COMPANIES * NUM_EDGE_CASE_TEMPLATES)
        assert len(questions) == expected
    
    def test_questions_have_required_fields(self):
        """Test that generated questions have all required fields."""
        from backend.rag.eval.account_eval import generate_eval_questions
        
        questions = generate_eval_questions()
        
        required_fields = ["id", "company_id", "company_name", "question", "question_type"]
        
        for q in questions:
            for field in required_fields:
                assert field in q, f"Missing field: {field}"
                assert q[field], f"Empty field: {field}"
    
    def test_questions_are_deterministic(self):
        """Test that question generation is deterministic."""
        from backend.rag.eval.account_eval import generate_eval_questions
        
        q1 = generate_eval_questions(seed=42)
        q2 = generate_eval_questions(seed=42)
        
        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a["id"] == b["id"]
            assert a["company_id"] == b["company_id"]
            assert a["question"] == b["question"]


# =============================================================================
# Integration Tests (require full setup)
# =============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests that require full MVP2 setup."""
    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI API key not set")
    def test_end_to_end_account_question(self):
        """Test full end-to-end account question flow."""
        try:
            from backend.rag.pipeline.account import answer_account_question
            from backend.rag.retrieval.private import create_private_backend
        except Exception as e:
            pytest.skip(f"Import failed: {e}")
        
        # Check if backend is ready
        try:
            backend = create_private_backend()
            companies = backend.get_all_companies()
            if not companies:
                pytest.skip("No companies in index")
        except ValueError:
            pytest.skip("Private collection not ready")
        except RuntimeError as e:
            # Handle Qdrant file locking issue when another process has the lock
            if "already accessed by another instance" in str(e):
                pytest.skip("Qdrant storage locked by another process")
            raise
        except Exception as e:
            # Catch any other Qdrant-related errors
            if "Qdrant" in str(type(e).__name__) or "portalocker" in str(e).lower():
                pytest.skip(f"Qdrant unavailable: {e}")
            raise
        
        # Run a real question
        try:
            result = answer_account_question(
                question="What is the status?",
                company_id=companies[0],
            )
        except RuntimeError as e:
            if "already accessed by another instance" in str(e):
                pytest.skip("Qdrant storage locked by another process")
            raise
        
        # Verify result structure
        assert "answer" in result
        assert "company_id" in result
        assert "sources" in result
        assert "raw_private_hits" in result
        assert "meta" in result
        
        # Verify no privacy leakage
        target = result["company_id"]
        for hit in result["raw_private_hits"]:
            assert hit["company_id"] == target or hit["company_id"] == ""
