"""
Tests for graph-level features: caching, per-node latencies.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from backend.agent.session.cache import (
    make_cache_key,
    get_cached_result,
    set_cached_result,
    clear_query_cache,
    _CACHE_TTL_SECONDS,
)


class TestQueryCache:
    """Tests for query caching functionality."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_query_cache()

    def test_make_cache_key_consistency(self):
        """Same inputs should produce same cache key."""
        key1 = make_cache_key("What is Acme?", "auto", "ACME-123")
        key2 = make_cache_key("What is Acme?", "auto", "ACME-123")
        assert key1 == key2

    def test_make_cache_key_case_insensitive(self):
        """Cache key should be case-insensitive for question."""
        key1 = make_cache_key("What is Acme?", "auto", None)
        key2 = make_cache_key("WHAT IS ACME?", "auto", None)
        assert key1 == key2

    def test_make_cache_key_strips_whitespace(self):
        """Cache key should strip leading/trailing whitespace."""
        key1 = make_cache_key("What is Acme?", "auto", None)
        key2 = make_cache_key("  What is Acme?  ", "auto", None)
        assert key1 == key2

    def test_make_cache_key_different_modes(self):
        """Different modes should produce different cache keys."""
        key1 = make_cache_key("What is Acme?", "auto", None)
        key2 = make_cache_key("What is Acme?", "docs", None)
        assert key1 != key2

    def test_make_cache_key_different_companies(self):
        """Different company IDs should produce different cache keys."""
        key1 = make_cache_key("What is Acme?", "auto", "ACME-123")
        key2 = make_cache_key("What is Acme?", "auto", "BETA-456")
        assert key1 != key2

    def test_cache_miss_returns_none(self):
        """Cache miss should return None."""
        result = get_cached_result("nonexistent-key")
        assert result is None

    def test_cache_set_and_get(self):
        """Set and get cache entry."""
        test_result = {"answer": "Test answer", "meta": {"mode_used": "docs"}}
        set_cached_result("test-key", test_result)

        cached = get_cached_result("test-key")
        assert cached == test_result

    def test_cache_clear(self):
        """Clear cache should remove all entries."""
        set_cached_result("key1", {"answer": "1"})
        set_cached_result("key2", {"answer": "2"})

        clear_query_cache()

        assert get_cached_result("key1") is None
        assert get_cached_result("key2") is None

    def test_cache_expiration(self):
        """Expired entries should not be returned."""
        test_result = {"answer": "Test"}
        set_cached_result("expire-key", test_result)

        # Manually expire the entry
        from backend.agent.session import cache
        cache._query_cache["expire-key"] = (test_result, time.time() - _CACHE_TTL_SECONDS - 1)

        assert get_cached_result("expire-key") is None


class TestMetaLatencies:
    """Tests for per-node latency tracking in meta response."""

    def test_meta_schema_has_latency_fields(self):
        """MetaInfo schema should have per-node latency fields."""
        from backend.agent.core.schemas import MetaInfo

        # Create a MetaInfo with all latencies
        meta = MetaInfo(
            mode_used="data",
            latency_ms=1000,
            router_latency_ms=50,
            fetch_latency_ms=200,
            answer_latency_ms=700,
            followup_latency_ms=50,
        )

        assert meta.router_latency_ms == 50
        assert meta.fetch_latency_ms == 200
        assert meta.answer_latency_ms == 700
        assert meta.followup_latency_ms == 50

    def test_meta_latencies_optional(self):
        """Per-node latency fields should be optional."""
        from backend.agent.core.schemas import MetaInfo

        # Create MetaInfo without latencies (should work)
        meta = MetaInfo(mode_used="docs", latency_ms=500)

        assert meta.router_latency_ms is None
        assert meta.fetch_latency_ms is None
        assert meta.answer_latency_ms is None
        assert meta.followup_latency_ms is None


class TestGraphExports:
    """Tests for graph module exports."""

    def test_exports_include_cache_functions(self):
        """Graph exports should include cache functions."""
        from backend.agent.graph import __all__

        assert "clear_query_cache" in __all__
