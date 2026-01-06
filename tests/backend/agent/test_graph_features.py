"""Tests for graph-level features: per-node latencies."""

import pytest


class TestMetaLatencies:
    """Tests for per-node latency tracking in meta response."""

    def test_meta_schema_has_latency_fields(self):
        """MetaInfo schema should have per-node latency fields."""
        from backend.agent.core.schemas import MetaInfo

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

        meta = MetaInfo(mode_used="docs", latency_ms=500)

        assert meta.router_latency_ms is None
        assert meta.fetch_latency_ms is None
        assert meta.answer_latency_ms is None
        assert meta.followup_latency_ms is None
