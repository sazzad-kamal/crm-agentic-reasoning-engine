"""Tests for CRM DataStore."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from backend.agent.datastore import (
    CRMDataStore,
    get_csv_base_path,
    CSV_TABLES,
    REQUIRED_TABLES,
)


class TestGetCsvBasePath:
    """Tests for get_csv_base_path function."""

    def test_returns_path(self):
        """Test returns a Path object."""
        path = get_csv_base_path()
        assert isinstance(path, Path)

    def test_path_exists(self):
        """Test returned path exists."""
        path = get_csv_base_path()
        assert path.exists()

    def test_path_is_directory(self):
        """Test returned path is a directory."""
        path = get_csv_base_path()
        assert path.is_dir()


class TestCsvTables:
    """Tests for CSV table configuration."""

    def test_csv_tables_defined(self):
        """Test CSV_TABLES has expected tables."""
        expected = ["companies", "contacts", "activities", "history", "opportunities"]
        for table in expected:
            assert table in CSV_TABLES

    def test_required_tables_subset_of_csv_tables(self):
        """Test required tables are in CSV_TABLES."""
        for table in REQUIRED_TABLES:
            assert table in CSV_TABLES


class TestCRMDataStoreInit:
    """Tests for CRMDataStore initialization."""

    def test_init_without_path(self):
        """Test init without explicit path uses auto-detection."""
        store = CRMDataStore()
        assert store._csv_path is None  # Lazy loaded

    def test_init_with_path(self):
        """Test init with explicit path."""
        test_path = Path("/tmp/test")
        store = CRMDataStore(csv_path=test_path)
        assert store._csv_path == test_path

    def test_lazy_connection(self):
        """Test connection is lazy-loaded."""
        store = CRMDataStore()
        assert store._conn is None  # Not created yet


class TestCRMDataStoreProperties:
    """Tests for CRMDataStore properties."""

    def test_csv_path_property(self):
        """Test csv_path property returns Path."""
        store = CRMDataStore()
        assert isinstance(store.csv_path, Path)

    def test_conn_property_creates_connection(self):
        """Test conn property creates DuckDB connection."""
        store = CRMDataStore()
        conn = store.conn
        assert conn is not None
        assert store._conn is not None


class TestCRMDataStoreResolveCompany:
    """Tests for resolve_company_id method."""

    def test_resolve_empty_returns_none(self):
        """Test empty input returns None."""
        store = CRMDataStore()
        assert store.resolve_company_id("") is None
        assert store.resolve_company_id(None) is None

    def test_resolve_by_id(self):
        """Test resolving by exact company ID."""
        store = CRMDataStore()
        # First call builds cache
        result = store.resolve_company_id("C001")
        # Either finds it or returns None, but should not error
        assert result is None or isinstance(result, str)

    def test_resolve_case_insensitive(self):
        """Test name matching is case-insensitive."""
        store = CRMDataStore()
        # The method should handle case-insensitive lookups
        result1 = store.resolve_company_id("acme")
        result2 = store.resolve_company_id("ACME")
        # Both should return same result (or None if not found)
        assert result1 == result2


class TestCRMDataStoreGetCompany:
    """Tests for get_company method."""

    def test_get_company_returns_dict_or_none(self):
        """Test get_company returns dict or None."""
        store = CRMDataStore()
        result = store.get_company("C001")
        assert result is None or isinstance(result, dict)

    def test_get_company_not_found(self):
        """Test get_company with invalid ID returns None."""
        store = CRMDataStore()
        result = store.get_company("INVALID_ID_XYZ123")
        assert result is None


class TestCRMDataStoreGetActivities:
    """Tests for get_recent_activities method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_recent_activities("C001", days=30, limit=10)
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_recent_activities("C001", days=90, limit=5)
        assert len(result) <= 5

    def test_returns_dicts(self):
        """Test returns list of dicts."""
        store = CRMDataStore()
        result = store.get_recent_activities("C001", days=90, limit=10)
        for item in result:
            assert isinstance(item, dict)


class TestCRMDataStoreGetHistory:
    """Tests for get_recent_history method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_recent_history("C001", days=30, limit=10)
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_recent_history("C001", days=90, limit=5)
        assert len(result) <= 5


class TestCRMDataStoreGetOpportunities:
    """Tests for get_open_opportunities method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_open_opportunities("C001", limit=10)
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_open_opportunities("C001", limit=3)
        assert len(result) <= 3


class TestCRMDataStoreGetPipelineSummary:
    """Tests for get_pipeline_summary method."""

    def test_returns_dict(self):
        """Test returns a dict."""
        store = CRMDataStore()
        result = store.get_pipeline_summary("C001")
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test result has expected keys."""
        store = CRMDataStore()
        result = store.get_pipeline_summary("C001")
        assert "stages" in result
        assert "total_count" in result
        assert "total_value" in result

    def test_stages_is_dict(self):
        """Test stages is a dict."""
        store = CRMDataStore()
        result = store.get_pipeline_summary("C001")
        assert isinstance(result["stages"], dict)


class TestCRMDataStoreGetRenewals:
    """Tests for get_upcoming_renewals method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_upcoming_renewals(days=90, limit=10)
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_upcoming_renewals(days=365, limit=5)
        assert len(result) <= 5

    def test_returns_dicts(self):
        """Test returns list of dicts."""
        store = CRMDataStore()
        result = store.get_upcoming_renewals(days=90, limit=10)
        for item in result:
            assert isinstance(item, dict)


class TestCRMDataStoreGetContacts:
    """Tests for get_contacts_for_company method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_contacts_for_company("C001", limit=10)
        assert isinstance(result, list)

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_contacts_for_company("C001", limit=3)
        assert len(result) <= 3


class TestCRMDataStoreSearchContacts:
    """Tests for search_contacts method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.search_contacts(query="john", limit=10)
        assert isinstance(result, list)

    def test_with_role_filter(self):
        """Test with role filter."""
        store = CRMDataStore()
        result = store.search_contacts(role="Decision Maker", limit=10)
        assert isinstance(result, list)

    def test_with_company_filter(self):
        """Test with company_id filter."""
        store = CRMDataStore()
        result = store.search_contacts(company_id="C001", limit=10)
        assert isinstance(result, list)


class TestCRMDataStoreSearchCompanies:
    """Tests for search_companies method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.search_companies(query="acme", limit=10)
        assert isinstance(result, list)

    def test_with_industry_filter(self):
        """Test with industry filter."""
        store = CRMDataStore()
        result = store.search_companies(industry="Technology", limit=10)
        assert isinstance(result, list)

    def test_with_status_filter(self):
        """Test with status filter."""
        store = CRMDataStore()
        result = store.search_companies(status="Active", limit=10)
        assert isinstance(result, list)

    def test_empty_query_returns_all(self):
        """Test empty query returns results."""
        store = CRMDataStore()
        result = store.search_companies(limit=5)
        assert isinstance(result, list)


class TestCRMDataStoreCompanyNameMatches:
    """Tests for get_company_name_matches method."""

    def test_returns_list(self):
        """Test returns a list."""
        store = CRMDataStore()
        result = store.get_company_name_matches("acme", limit=5)
        assert isinstance(result, list)

    def test_empty_query_returns_empty(self):
        """Test empty query returns empty list."""
        store = CRMDataStore()
        result = store.get_company_name_matches("", limit=5)
        assert result == []

    def test_respects_limit(self):
        """Test respects limit parameter."""
        store = CRMDataStore()
        result = store.get_company_name_matches("tech", limit=3)
        assert len(result) <= 3
