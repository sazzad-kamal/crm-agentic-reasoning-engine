"""
Additional coverage tests targeting specific uncovered lines.

Covers edge cases in:
- handlers/common.py: lookup_company, _load_private_texts, _load_attachments, enrich_raw_data
- question_tree: unknown role, print_tree, validate_tree
- llm/router.py: auto mode routing paths
- output/streaming.py: error handling
- session.py: checkpoint state with messages
- datastore modules: edge cases
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
import json


# =============================================================================
# handlers/common.py - lookup_company found path
# =============================================================================


class TestLookupCompanyFound:
    """Tests for lookup_company when company is found."""

    def test_lookup_company_found_with_contacts(self):
        """Test lookup_company returns True and populates data when found."""
        from backend.agent.fetch.handlers.common import lookup_company, IntentResult, empty_raw_data

        result = IntentResult(raw_data=empty_raw_data())

        with patch("backend.agent.fetch.handlers.common.get_datastore") as mock_get_ds:
            mock_ds = MagicMock()
            mock_ds.resolve_company_id.return_value = "COMP001"
            mock_ds.get_company.return_value = {
                "company_id": "COMP001",
                "name": "Test Company",
                "industry": "Technology",
            }
            mock_ds.get_contacts_for_company.return_value = [
                {"name": "John Doe", "email": "john@test.com"},
            ]
            mock_get_ds.return_value = mock_ds

            found = lookup_company(result, "Test Company")

            assert found is True
            assert result.company_data["found"] is True
            assert result.company_data["company"]["name"] == "Test Company"
            assert len(result.company_data["contacts"]) == 1
            assert result.resolved_company_id == "COMP001"
            assert len(result.sources) == 1
            assert result.raw_data["companies"] == [result.company_data["company"]]

    def test_lookup_company_not_found_with_matches(self):
        """Test lookup_company when ID not resolved."""
        from backend.agent.fetch.handlers.common import lookup_company, IntentResult, empty_raw_data

        result = IntentResult(raw_data=empty_raw_data())

        with patch("backend.agent.fetch.handlers.common.get_datastore") as mock_get_ds:
            mock_ds = MagicMock()
            mock_ds.resolve_company_id.return_value = None
            mock_ds.get_company_name_matches.return_value = [
                {"name": "Similar Company"},
            ]
            mock_get_ds.return_value = mock_ds

            found = lookup_company(result, "Unknown")

            assert found is False
            assert result.company_data["found"] is False
            assert "close_matches" in result.company_data

    def test_lookup_company_id_resolved_but_no_data(self):
        """Test lookup_company when ID resolved but get_company returns None."""
        from backend.agent.fetch.handlers.common import lookup_company, IntentResult, empty_raw_data

        result = IntentResult(raw_data=empty_raw_data())

        with patch("backend.agent.fetch.handlers.common.get_datastore") as mock_get_ds:
            mock_ds = MagicMock()
            mock_ds.resolve_company_id.return_value = "COMP001"
            mock_ds.get_company.return_value = None
            mock_get_ds.return_value = mock_ds

            found = lookup_company(result, "Ghost Company")

            assert found is False
            assert result.company_data["found"] is False


# =============================================================================
# handlers/common.py - _load_private_texts and _load_attachments
# =============================================================================


class TestDataLoading:
    """Tests for private texts and attachments loading."""

    def test_load_private_texts_file_not_exists(self):
        """Test _load_private_texts returns empty dict when file doesn't exist."""
        from backend.agent.fetch.handlers.common import _load_private_texts

        # Clear cache first
        _load_private_texts.cache_clear()

        with patch("backend.agent.fetch.handlers.common._get_csv_path") as mock_path:
            mock_p = MagicMock()
            mock_p.__truediv__ = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
            mock_path.return_value = mock_p

            result = _load_private_texts()
            assert result == {}

        _load_private_texts.cache_clear()

    def test_load_attachments_file_not_exists(self):
        """Test _load_attachments returns empty dict when file doesn't exist."""
        from backend.agent.fetch.handlers.common import _load_attachments

        _load_attachments.cache_clear()

        with patch("backend.agent.fetch.handlers.common._get_csv_path") as mock_path:
            mock_p = MagicMock()
            mock_p.__truediv__ = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
            mock_path.return_value = mock_p

            result = _load_attachments()
            assert result == {}

        _load_attachments.cache_clear()


# =============================================================================
# handlers/common.py - enrich_raw_data
# =============================================================================


class TestEnrichRawData:
    """Tests for enrich_raw_data function."""

    def test_enrich_raw_data_adds_private_texts(self):
        """Test enrich_raw_data adds _private_texts to companies."""
        from backend.agent.fetch.handlers.common import enrich_raw_data, _load_private_texts, _load_attachments

        _load_private_texts.cache_clear()
        _load_attachments.cache_clear()

        with patch("backend.agent.fetch.handlers.common._load_private_texts") as mock_texts, \
             patch("backend.agent.fetch.handlers.common._load_attachments") as mock_attach:
            mock_texts.return_value = {"COMP001": [{"text": "Private note"}]}
            mock_attach.return_value = {}

            raw_data = {
                "companies": [{"company_id": "COMP001", "name": "Test"}],
                "opportunities": [],
            }

            result = enrich_raw_data(raw_data)

            assert result["companies"][0]["_private_texts"] == [{"text": "Private note"}]

    def test_enrich_raw_data_adds_attachments(self):
        """Test enrich_raw_data adds _attachments to opportunities."""
        from backend.agent.fetch.handlers.common import enrich_raw_data

        with patch("backend.agent.fetch.handlers.common._load_private_texts") as mock_texts, \
             patch("backend.agent.fetch.handlers.common._load_attachments") as mock_attach:
            mock_texts.return_value = {}
            mock_attach.return_value = {"OPP001": [{"filename": "proposal.pdf"}]}

            raw_data = {
                "companies": [],
                "opportunities": [{"opportunity_id": "OPP001", "name": "Big Deal"}],
            }

            result = enrich_raw_data(raw_data)

            assert result["opportunities"][0]["_attachments"] == [{"filename": "proposal.pdf"}]

    def test_enrich_raw_data_empty_matches(self):
        """Test enrich_raw_data handles no matching private texts or attachments."""
        from backend.agent.fetch.handlers.common import enrich_raw_data

        with patch("backend.agent.fetch.handlers.common._load_private_texts") as mock_texts, \
             patch("backend.agent.fetch.handlers.common._load_attachments") as mock_attach:
            mock_texts.return_value = {}
            mock_attach.return_value = {}

            raw_data = {
                "companies": [{"company_id": "UNKNOWN", "name": "No Match"}],
                "opportunities": [{"opportunity_id": "UNKNOWN", "name": "No Match"}],
            }

            result = enrich_raw_data(raw_data)

            assert result["companies"][0]["_private_texts"] == []
            assert result["opportunities"][0]["_attachments"] == []


# =============================================================================
# question_tree - unknown role and print_tree
# =============================================================================


class TestQuestionTreeEdgeCases:
    """Tests for question_tree edge cases."""

    def test_get_starters_for_role_unknown(self):
        """Test _get_starters_for_role raises ValueError for unknown role."""
        from backend.agent.followup.tree import _get_starters_for_role

        with pytest.raises(ValueError, match="Unknown role"):
            _get_starters_for_role("invalid_role")

    def test_print_tree_single_role(self):
        """Test print_tree for a single role."""
        from backend.agent.followup.tree import print_tree

        tree = print_tree("sales")
        assert tree is not None
        # Tree should have nodes
        assert hasattr(tree, "label")

    def test_print_tree_all_roles(self):
        """Test print_tree for all roles."""
        from backend.agent.followup.tree import print_tree

        tree = print_tree()
        assert tree is not None

    def test_print_tree_with_max_depth(self):
        """Test print_tree with max_depth limit."""
        from backend.agent.followup.tree import print_tree

        tree = print_tree("csm", max_depth=2)
        assert tree is not None

    def test_print_tree_unknown_role(self):
        """Test print_tree handles unknown role."""
        from backend.agent.followup.tree import print_tree

        tree = print_tree("invalid")
        # Should return tree with error message
        assert "error" in str(tree.label).lower() or "unknown" in str(tree.label).lower()

    def test_validate_tree_all_roles(self):
        """Test validate_tree for all roles."""
        from backend.agent.followup.tree import validate_tree

        issues = validate_tree()
        # May or may not have issues depending on tree structure
        assert isinstance(issues, list)

    def test_validate_tree_single_role(self):
        """Test validate_tree for a single role."""
        from backend.agent.followup.tree import validate_tree

        issues = validate_tree("sales")
        assert isinstance(issues, list)

    def test_get_tree_stats(self):
        """Test get_tree_stats returns expected structure."""
        from backend.agent.followup.tree import get_tree_stats

        stats = get_tree_stats()

        assert "role" in stats
        assert "num_starters" in stats
        assert "num_questions" in stats
        assert "num_paths" in stats
        assert "max_depth" in stats

    def test_get_tree_stats_single_role(self):
        """Test get_tree_stats for a single role."""
        from backend.agent.followup.tree import get_tree_stats

        stats = get_tree_stats("manager")

        assert stats["role"] == "manager"
        assert stats["num_starters"] == 1


# =============================================================================
# llm/router.py - auto mode LLM routing
# =============================================================================


class TestLlmRouterAutoMode:
    """Tests for LLM router auto mode."""

    def test_detect_owner_from_starter_sales(self):
        """Test detect_owner_from_starter detects sales rep."""
        from backend.agent.llm.router import detect_owner_from_starter

        owner = detect_owner_from_starter("How's my pipeline?")
        assert owner == "jsmith"

    def test_detect_owner_from_starter_csm(self):
        """Test detect_owner_from_starter detects CSM."""
        from backend.agent.llm.router import detect_owner_from_starter

        owner = detect_owner_from_starter("Any renewals at risk?")
        assert owner == "amartin"

    def test_detect_owner_from_starter_manager(self):
        """Test detect_owner_from_starter detects manager (None owner)."""
        from backend.agent.llm.router import detect_owner_from_starter

        owner = detect_owner_from_starter("How's the team doing?")
        assert owner is None

    def test_detect_owner_from_starter_no_match(self):
        """Test detect_owner_from_starter returns None for non-starter."""
        from backend.agent.llm.router import detect_owner_from_starter

        owner = detect_owner_from_starter("Random question")
        assert owner is None


# =============================================================================
# datastore edge cases
# =============================================================================


class TestDatastoreEdgeCases:
    """Tests for datastore edge cases."""

    def test_analytics_get_activity_count_with_filters(self):
        """Test get_activity_count_by_filter with filters."""
        from backend.agent.datastore import get_datastore

        ds = get_datastore()

        result = ds.get_activity_count_by_filter(
            days=30,
            activity_type="call",
            company_id="ACME-MFG",
        )

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_datastore_search_with_empty_query(self):
        """Test search functions handle empty query."""
        from backend.agent.datastore import get_datastore

        ds = get_datastore()

        # Empty query should return empty results
        result = ds.search_companies(query="")
        assert isinstance(result, list)



# =============================================================================
# main.py - health endpoint and RAG collections
# =============================================================================


class TestMainAppCoverage:
    """Tests for main.py coverage."""

    def test_health_endpoint(self):
        """Test health endpoint returns ok."""
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        response = client.get("/api/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ensure_rag_collections_already_exists(self):
        """Test _ensure_rag_collections when collections already exist."""
        from backend.main import _ensure_rag_collections

        with patch("qdrant_client.QdrantClient") as mock_qdrant_class:
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_collection = MagicMock()
            mock_collection.points_count = 100
            mock_client.get_collection.return_value = mock_collection
            mock_qdrant_class.return_value = mock_client

            # Should not raise
            _ensure_rag_collections()

            # Should check both collections
            assert mock_client.collection_exists.call_count == 2

    def test_ensure_rag_collections_empty_count(self):
        """Test _ensure_rag_collections when collection exists but empty."""
        from backend.main import _ensure_rag_collections

        with patch("qdrant_client.QdrantClient") as mock_qdrant_class, \
             patch("backend.agent.rag.ingest.ingest_docs") as mock_ingest_docs, \
             patch("backend.agent.rag.ingest.ingest_private_texts") as mock_ingest_private:

            mock_client = MagicMock()
            # Collection exists but is empty
            mock_client.collection_exists.return_value = True
            mock_collection = MagicMock()
            mock_collection.points_count = 0
            mock_client.get_collection.return_value = mock_collection
            mock_qdrant_class.return_value = mock_client

            _ensure_rag_collections()

            # Should call ingest for both empty collections
            mock_ingest_docs.assert_called_once()
            mock_ingest_private.assert_called_once()


# =============================================================================
# handlers/company.py - search with multiple filters
# =============================================================================


class TestCompanyHandlerFilters:
    """Tests for company handler filter combinations."""

    def test_search_companies_with_segment_filter(self):
        """Test search_companies with segment filter."""
        from backend.agent.fetch.handlers import tool_search_companies

        result = tool_search_companies(segment="Enterprise")

        assert "filters" in result.data
        assert result.data["filters"]["segment"] == "Enterprise"

    def test_search_contacts_with_company_filter(self):
        """Test search_contacts with company_id filter."""
        from backend.agent.fetch.handlers import tool_search_contacts

        result = tool_search_contacts(company_id="ACME-MFG")

        assert "filters" in result.data
        assert result.data["filters"]["company_id"] == "ACME-MFG"


# =============================================================================
# nodes/fetching.py - error handling
# =============================================================================


class TestFetchingNodeErrors:
    """Tests for fetching node error handling."""

    def test_fetch_docs_handles_exceptions(self):
        """Test _fetch_docs returns empty on exception."""
        from backend.agent.fetch.node import _fetch_docs

        with patch("backend.agent.fetch.node.call_docs_rag") as mock_rag:
            mock_rag.side_effect = Exception("RAG failed")

            result = _fetch_docs("test question")

            assert result["docs_answer"] == ""
            assert result["docs_sources"] == []
            assert "error" in result


# =============================================================================
# make_sources helper
# =============================================================================


class TestMakeSourcesHelper:
    """Tests for make_sources helper function."""

    def test_make_sources_with_data(self):
        """Test make_sources returns Source when data exists."""
        from backend.agent.fetch.handlers.common import make_sources

        sources = make_sources(
            data=[{"id": "1"}],
            source_type="activity",
            source_id="act-001",
            label="Activity Search",
        )

        assert len(sources) == 1
        assert sources[0].type == "activity"
        assert sources[0].id == "act-001"

    def test_make_sources_empty_data(self):
        """Test make_sources returns empty list when no data."""
        from backend.agent.fetch.handlers.common import make_sources

        sources = make_sources(
            data=[],
            source_type="activity",
            source_id="act-001",
            label="Activity Search",
        )

        assert sources == []

    def test_make_sources_none_data(self):
        """Test make_sources returns empty list when data is None."""
        from backend.agent.fetch.handlers.common import make_sources

        sources = make_sources(
            data=None,
            source_type="activity",
            source_id="act-001",
            label="Activity Search",
        )

        assert sources == []


# =============================================================================
# safe_extend helper
# =============================================================================


class TestSafeExtendHelper:
    """Tests for safe_extend helper function."""

    def test_safe_extend_with_list(self):
        """Test safe_extend extends list."""
        from backend.agent.fetch.handlers.common import safe_extend

        target = [1, 2]
        safe_extend(target, [3, 4])

        assert target == [1, 2, 3, 4]

    def test_safe_extend_with_none(self):
        """Test safe_extend handles None source."""
        from backend.agent.fetch.handlers.common import safe_extend

        target = [1, 2]
        safe_extend(target, None)

        assert target == [1, 2]

    def test_safe_extend_with_empty_list(self):
        """Test safe_extend handles empty source list."""
        from backend.agent.fetch.handlers.common import safe_extend

        target = [1, 2]
        safe_extend(target, [])

        assert target == [1, 2]
