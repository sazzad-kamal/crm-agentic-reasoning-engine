"""
Tests for RAG modules (client, ingest, tools).

These tests mock external dependencies (Qdrant, LlamaIndex) to achieve coverage
without requiring actual infrastructure.
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


class TestRagClient:
    """Tests for backend.agent.rag.client module."""

    def test_get_qdrant_client_creates_singleton(self):
        """Test that get_qdrant_client returns a singleton."""
        from backend.agent.fetch.rag import client

        # Reset singleton
        client._qdrant_client = None

        with patch.object(client, "QdrantClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # First call creates client
            result1 = client.get_qdrant_client()
            assert result1 is mock_instance

            # Second call returns same client (singleton)
            result2 = client.get_qdrant_client()
            assert result2 is result1

            # QdrantClient should only be called once
            assert mock_client.call_count == 1

        # Cleanup
        client._qdrant_client = None

    def test_close_qdrant_client(self):
        """Test that close_qdrant_client closes and clears singleton."""
        from backend.agent.fetch.rag import client

        mock_instance = MagicMock()
        client._qdrant_client = mock_instance

        client.close_qdrant_client()

        mock_instance.close.assert_called_once()
        assert client._qdrant_client is None

    def test_close_qdrant_client_when_none(self):
        """Test close_qdrant_client when no client exists."""
        from backend.agent.fetch.rag import client

        client._qdrant_client = None
        # Should not raise
        client.close_qdrant_client()
        assert client._qdrant_client is None


class TestRagIngest:
    """Tests for backend.agent.rag.ingest module."""

    def test_ingest_texts_no_file(self):
        """Test ingest_texts returns 0 when JSONL doesn't exist."""
        # This is a simpler test that just checks the file-not-exists path
        # without needing to mock llama_index since it returns early
        from backend.agent.fetch.rag.ingest import ingest_texts

        with patch("backend.agent.fetch.rag.ingest.JSONL_PATH") as mock_path:
            mock_path.exists.return_value = False
            result = ingest_texts()
            assert result == 0

    def test_jsonl_path_configured(self):
        """Test JSONL_PATH is properly configured."""
        from backend.agent.fetch.rag.config import JSONL_PATH
        assert JSONL_PATH is not None
        assert str(JSONL_PATH).endswith(".jsonl")


class TestLlmClient:
    """Tests for backend.core.llm module."""

    def test_get_chat_model_no_api_key(self):
        """Test get_chat_model raises when no API key."""
        from backend.core.llm import get_chat_model

        # Clear cache
        get_chat_model.cache_clear()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_chat_model("gpt-4o-mini", 0.0, 1024)

    def test_get_chat_model_with_api_key(self):
        """Test get_chat_model creates ChatOpenAI with API key."""
        from backend.core.llm import get_chat_model

        # Clear cache
        get_chat_model.cache_clear()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch("backend.core.llm.ChatOpenAI") as mock_chat:

            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            result = get_chat_model("gpt-4o-mini", 0.0, 1024)

            assert result is mock_instance
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "gpt-4o-mini"
            assert call_kwargs["temperature"] == 0.0
            assert call_kwargs["max_tokens"] == 1024

        # Clear cache for other tests
        get_chat_model.cache_clear()

    def test_get_chat_model_o1_uses_max_completion_tokens(self):
        """Test that o1 models use max_completion_tokens parameter."""
        from backend.core.llm import get_chat_model

        get_chat_model.cache_clear()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch("backend.core.llm.ChatOpenAI") as mock_chat:

            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            get_chat_model("o1-preview", 0.0, 1024)

            call_kwargs = mock_chat.call_args[1]
            assert "max_completion_tokens" in call_kwargs
            assert call_kwargs["max_completion_tokens"] == 1024
            assert "max_tokens" not in call_kwargs
            assert "temperature" not in call_kwargs

        get_chat_model.cache_clear()

    def test_call_llm_with_system_prompt(self):
        """Test call_llm with system prompt."""
        from backend.core.llm import call_llm, get_chat_model

        get_chat_model.cache_clear()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch("backend.core.llm.ChatOpenAI") as mock_chat:

            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_response
            mock_chat.return_value = mock_instance

            result = call_llm("Test prompt", system_prompt="System context")

            assert result == "Test response"
            # Should have 2 messages (system + human)
            call_args = mock_instance.invoke.call_args[0][0]
            assert len(call_args) == 2

        get_chat_model.cache_clear()

    def test_call_llm_without_system_prompt(self):
        """Test call_llm without system prompt."""
        from backend.core.llm import call_llm, get_chat_model

        get_chat_model.cache_clear()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch("backend.core.llm.ChatOpenAI") as mock_chat:

            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = mock_response
            mock_chat.return_value = mock_instance

            result = call_llm("Test prompt")

            assert result == "Test response"
            # Should have 1 message (human only)
            call_args = mock_instance.invoke.call_args[0][0]
            assert len(call_args) == 1

        get_chat_model.cache_clear()


class TestEvalFormatting:
    """Tests for uncovered formatting functions."""

    def test_build_eval_table(self):
        """Test build_eval_table creates table with sections."""
        from backend.eval.shared.formatting import build_eval_table

        sections = [
            ("Section1", [
                ("Metric1", "100", "90", True),
                ("Metric2", "80", "90", False),
            ]),
            ("Section2", [
                ("Metric3", "50", None, None),
            ]),
        ]

        table = build_eval_table("Test Table", sections)

        assert table.title == "Test Table"
        assert len(table.columns) == 3  # Metric, Value, SLO
