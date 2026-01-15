"""
Tests for reranker configuration.
"""


class TestRerankerConfig:
    """Tests for reranker configuration."""

    def test_reranker_config_exists(self):
        """Test that reranker config values are present."""
        from backend.agent.fetch.rag.config import (
            RERANKER_MODEL,
            RERANKER_TOP_K,
            SEARCH_TOP_K,
        )

        assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert RERANKER_TOP_K == 5
        assert SEARCH_TOP_K == 30
