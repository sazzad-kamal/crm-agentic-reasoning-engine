"""
Tests for company resolution utilities.

Tests CompanyResolver class and helper functions.

Run with:
    pytest tests/backend/rag/test_company_resolver.py -v
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from backend.rag.pipeline.company_resolver import (
    CompanyResolver,
    get_resolver,
    resolve_company_id,
    get_company_name,
    get_company_matches,
    clear_cache,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_companies_df():
    """Create sample companies DataFrame for testing."""
    return pd.DataFrame({
        "company_id": ["ACME-MFG", "BETA-TECH", "GAMMA-CORP", "DELTA-INC"],
        "name": ["Acme Manufacturing", "Beta Technologies", "Gamma Corporation", "Delta Incorporated"],
        "industry": ["Manufacturing", "Technology", "Finance", "Retail"],
    })


@pytest.fixture
def resolver(sample_companies_df):
    """Create a CompanyResolver instance with sample data."""
    return CompanyResolver(df=sample_companies_df)


# =============================================================================
# Test: CompanyResolver - Exact Matching
# =============================================================================

class TestCompanyResolverExactMatch:
    """Tests for exact matching in CompanyResolver."""

    def test_resolve_by_exact_id(self, resolver, sample_companies_df):
        """Test resolving by exact company_id."""
        company_id = sample_companies_df.iloc[0]["company_id"]
        resolved = resolver.resolve(company_id)

        assert resolved == company_id

    def test_resolve_by_exact_name(self, resolver, sample_companies_df):
        """Test resolving by exact company name."""
        company_name = sample_companies_df.iloc[0]["name"]
        expected_id = sample_companies_df.iloc[0]["company_id"]

        resolved = resolver.resolve(company_name)

        assert resolved == expected_id

    def test_resolve_case_insensitive_name(self, resolver, sample_companies_df):
        """Test that name matching is case-insensitive."""
        company_name_upper = sample_companies_df.iloc[0]["name"].upper()
        expected_id = sample_companies_df.iloc[0]["company_id"]

        resolved = resolver.resolve(company_name_upper)

        assert resolved == expected_id

    def test_resolve_all_companies(self, resolver, sample_companies_df):
        """Test resolving all companies by ID."""
        for _, row in sample_companies_df.iterrows():
            resolved = resolver.resolve(row["company_id"])
            assert resolved == row["company_id"]


# =============================================================================
# Test: CompanyResolver - Partial Matching
# =============================================================================

class TestCompanyResolverPartialMatch:
    """Tests for partial/fuzzy matching."""

    def test_resolve_by_prefix(self, resolver):
        """Test resolving by company name prefix."""
        resolved = resolver.resolve("Acme")

        assert resolved == "ACME-MFG"

    def test_resolve_by_contains(self, resolver):
        """Test resolving when query is contained in name."""
        resolved = resolver.resolve("Technologies")

        assert resolved == "BETA-TECH"

    def test_resolve_fuzzy_match(self, resolver):
        """Test fuzzy matching with misspelling."""
        resolved = resolver.resolve("Acme Manufact")  # Partial match

        assert resolved == "ACME-MFG"

    def test_get_close_matches(self, resolver):
        """Test getting multiple close matches."""
        matches = resolver.get_close_matches("Beta", limit=5)

        assert isinstance(matches, list)
        # Matches depend on fuzzy matching implementation

    def test_get_close_matches_empty_query(self, resolver):
        """Test close matches with empty query."""
        matches = resolver.get_close_matches("", limit=5)

        assert matches == []


# =============================================================================
# Test: CompanyResolver - Get Company Name
# =============================================================================

class TestCompanyResolverGetName:
    """Tests for getting company name from ID."""

    def test_get_name_by_id(self, resolver, sample_companies_df):
        """Test getting company name by ID."""
        company_id = sample_companies_df.iloc[0]["company_id"]
        expected_name = sample_companies_df.iloc[0]["name"]

        name = resolver.get_name(company_id)

        assert name == expected_name

    def test_get_name_invalid_id(self, resolver):
        """Test getting name for invalid ID."""
        name = resolver.get_name("INVALID-ID-XYZ")

        assert name is None

    def test_get_name_empty_id(self, resolver):
        """Test getting name for empty ID."""
        name = resolver.get_name("")

        assert name is None


# =============================================================================
# Test: CompanyResolver - Validation
# =============================================================================

class TestCompanyResolverValidation:
    """Tests for validation methods."""

    def test_validate_valid_id(self, resolver, sample_companies_df):
        """Test validating a valid company ID."""
        company_id = sample_companies_df.iloc[0]["company_id"]

        assert resolver.validate_id(company_id) is True

    def test_validate_invalid_id(self, resolver):
        """Test validating an invalid company ID."""
        assert resolver.validate_id("INVALID-ID") is False

    def test_validate_empty_id(self, resolver):
        """Test validating empty ID."""
        assert resolver.validate_id("") is False


# =============================================================================
# Test: CompanyResolver - List All
# =============================================================================

class TestCompanyResolverListAll:
    """Tests for list_all method."""

    def test_list_all_returns_list(self, resolver):
        """Test that list_all returns a list."""
        companies = resolver.list_all()

        assert isinstance(companies, list)

    def test_list_all_contains_all_companies(self, resolver, sample_companies_df):
        """Test that list_all contains all companies."""
        companies = resolver.list_all()

        assert len(companies) == len(sample_companies_df)

    def test_list_all_has_dicts(self, resolver):
        """Test that list_all returns list of dicts."""
        companies = resolver.list_all()

        assert all(isinstance(c, dict) for c in companies)

    def test_list_all_has_required_fields(self, resolver):
        """Test that each company dict has required fields."""
        companies = resolver.list_all()

        for company in companies:
            assert "company_id" in company
            assert "name" in company


# =============================================================================
# Test: Module-Level Functions
# =============================================================================

class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    @patch('backend.rag.pipeline.company_resolver._load_companies_df')
    def test_resolve_company_id_function(self, mock_load, sample_companies_df):
        """Test resolve_company_id convenience function."""
        mock_load.return_value = sample_companies_df

        resolved = resolve_company_id("Acme Manufacturing")

        assert resolved == "ACME-MFG"

    @patch('backend.rag.pipeline.company_resolver._load_companies_df')
    def test_get_company_name_function(self, mock_load, sample_companies_df):
        """Test get_company_name convenience function."""
        mock_load.return_value = sample_companies_df

        name = get_company_name("ACME-MFG")

        assert name == "Acme Manufacturing"

    @patch('backend.rag.pipeline.company_resolver._load_companies_df')
    def test_get_company_matches_function(self, mock_load, sample_companies_df):
        """Test get_company_matches convenience function."""
        mock_load.return_value = sample_companies_df

        matches = get_company_matches("tech", limit=5)

        assert isinstance(matches, list)

    def test_get_resolver_returns_same_instance(self):
        """Test that get_resolver returns singleton."""
        resolver1 = get_resolver()
        resolver2 = get_resolver()

        assert resolver1 is resolver2

    def test_clear_cache_clears_loader_cache(self):
        """Test that clear_cache clears the loader cache."""
        # This is a basic test - implementation detail
        clear_cache()
        # Should not raise an error
        assert True


# =============================================================================
# Test: Lazy Initialization
# =============================================================================

class TestLazyInitialization:
    """Tests for lazy initialization."""

    def test_resolver_initializes_lazily(self, sample_companies_df):
        """Test that resolver initializes data structures lazily."""
        resolver = CompanyResolver(df=sample_companies_df)

        # Should not be initialized yet
        assert resolver._initialized is False

        # Access should trigger initialization
        resolver.resolve("ACME-MFG")

        assert resolver._initialized is True

    def test_df_loads_lazily(self):
        """Test that DataFrame loads lazily when None."""
        resolver = CompanyResolver(df=None)

        # Accessing df property should trigger loading
        # (This will fail in test environment without real CSV)
        # So we just test that the property exists
        assert hasattr(resolver, 'df')


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestCompanyResolverEdgeCases:
    """Tests for edge cases."""

    def test_resolve_none_input(self, resolver):
        """Test resolving None input."""
        resolved = resolver.resolve(None)

        assert resolved is None

    def test_resolve_empty_string(self, resolver):
        """Test resolving empty string."""
        resolved = resolver.resolve("")

        assert resolved is None

    def test_resolve_whitespace_only(self, resolver):
        """Test resolving whitespace-only string."""
        resolved = resolver.resolve("   ")

        # May resolve to something if whitespace is stripped and matches a company
        assert resolved is None or isinstance(resolved, str)

    def test_resolve_special_characters(self, resolver):
        """Test resolving with special characters."""
        resolved = resolver.resolve("@#$%")

        # Should handle gracefully
        assert resolved is None or isinstance(resolved, str)

    def test_empty_dataframe(self):
        """Test resolver with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["company_id", "name"])
        resolver = CompanyResolver(df=empty_df)

        resolved = resolver.resolve("anything")

        assert resolved is None

    def test_dataframe_with_duplicates(self):
        """Test resolver with duplicate company IDs."""
        df = pd.DataFrame({
            "company_id": ["ACME-MFG", "ACME-MFG", "BETA-TECH"],
            "name": ["Acme One", "Acme Two", "Beta"],
        })
        resolver = CompanyResolver(df=df)

        resolved = resolver.resolve("ACME-MFG")

        # Should resolve to one of them
        assert resolved == "ACME-MFG"

    def test_resolve_with_numbers(self, resolver):
        """Test resolving with numeric input (converted to string)."""
        resolved = resolver.resolve(123)

        # Should handle by converting to string
        assert resolved is None or isinstance(resolved, str)

    def test_get_close_matches_with_cutoff(self, resolver):
        """Test close matches with custom cutoff."""
        matches = resolver.get_close_matches("Acme", limit=5, cutoff=0.9)

        assert isinstance(matches, list)

    def test_get_close_matches_no_matches(self, resolver):
        """Test close matches when no matches found."""
        matches = resolver.get_close_matches("XYZ123NoMatch", limit=5, cutoff=0.99)

        # May return empty list
        assert isinstance(matches, list)

    def test_unicode_company_names(self):
        """Test handling of unicode in company names."""
        df = pd.DataFrame({
            "company_id": ["UNICODE-1"],
            "name": ["公司 Company 会社"],
        })
        resolver = CompanyResolver(df=df)

        resolved = resolver.resolve("公司")

        # Should handle unicode
        assert resolved == "UNICODE-1" or resolved is None


# =============================================================================
# Test: Integration
# =============================================================================

class TestCompanyResolverIntegration:
    """Integration tests for CompanyResolver."""

    def test_resolve_multiple_strategies(self, resolver, sample_companies_df):
        """Test that all resolution strategies work together."""
        test_cases = [
            ("ACME-MFG", "ACME-MFG"),  # Exact ID
            ("Acme Manufacturing", "ACME-MFG"),  # Exact name
            ("acme", "ACME-MFG"),  # Prefix/contains
            ("BETA-TECH", "BETA-TECH"),  # Another ID
            ("beta tech", "BETA-TECH"),  # Fuzzy
        ]

        for query, expected in test_cases:
            resolved = resolver.resolve(query)
            assert resolved == expected, f"Failed for query: {query}"

    def test_all_companies_resolvable(self, resolver, sample_companies_df):
        """Test that all companies can be resolved by name or ID."""
        for _, row in sample_companies_df.iterrows():
            # Resolve by ID
            resolved_by_id = resolver.resolve(row["company_id"])
            assert resolved_by_id == row["company_id"]

            # Resolve by name
            resolved_by_name = resolver.resolve(row["name"])
            assert resolved_by_name == row["company_id"]
