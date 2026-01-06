"""
Tests for RAG modules (client, ingest, tools).

These tests mock external dependencies (Qdrant, LlamaIndex) to achieve coverage
without requiring actual infrastructure.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


class TestRagClient:
    """Tests for backend.agent.rag.client module."""

    def test_get_qdrant_client_creates_singleton(self):
        """Test that get_qdrant_client returns a singleton."""
        from backend.agent.rag import client

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
        from backend.agent.rag import client

        mock_instance = MagicMock()
        client._qdrant_client = mock_instance

        client.close_qdrant_client()

        mock_instance.close.assert_called_once()
        assert client._qdrant_client is None

    def test_close_qdrant_client_when_none(self):
        """Test close_qdrant_client when no client exists."""
        from backend.agent.rag import client

        client._qdrant_client = None
        # Should not raise
        client.close_qdrant_client()
        assert client._qdrant_client is None

    def test_get_embed_model_lazy_load(self):
        """Test that embed model is lazily loaded."""
        from backend.agent.rag import client

        # Reset
        client._embed_model = None

        with patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding") as mock_embed:
            mock_model = MagicMock()
            mock_embed.return_value = mock_model

            result = client._get_embed_model()
            assert result is mock_model

            # Second call returns cached model
            result2 = client._get_embed_model()
            assert result2 is result
            assert mock_embed.call_count == 1

        # Cleanup
        client._embed_model = None

    def test_get_docs_index_creates_singleton(self):
        """Test that get_docs_index creates and caches index."""
        from backend.agent.rag import client

        # Reset
        client._docs_index = None
        client._qdrant_client = None

        with patch.object(client, "get_qdrant_client") as mock_get_client, \
             patch.object(client, "_get_embed_model") as mock_get_embed, \
             patch("llama_index.core.VectorStoreIndex") as mock_index_class, \
             patch("llama_index.vector_stores.qdrant.QdrantVectorStore") as mock_store_class, \
             patch("llama_index.core.Settings") as mock_settings:

            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            mock_embed = MagicMock()
            mock_get_embed.return_value = mock_embed

            mock_store = MagicMock()
            mock_store_class.return_value = mock_store

            mock_index = MagicMock()
            mock_index_class.from_vector_store.return_value = mock_index

            result = client.get_docs_index()
            assert result is mock_index

            # Second call returns cached index
            result2 = client.get_docs_index()
            assert result2 is result
            assert mock_index_class.from_vector_store.call_count == 1

        # Cleanup
        client._docs_index = None


class TestRagIngest:
    """Tests for backend.agent.rag.ingest module."""

    def test_ingest_docs_no_directory(self):
        """Test ingest_docs returns 0 when docs directory doesn't exist."""
        with patch("backend.agent.rag.ingest.DOCS_DIR") as mock_dir, \
             patch("llama_index.core.Settings"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.core.node_parser.SentenceSplitter"):
            mock_dir.exists.return_value = False

            from backend.agent.rag.ingest import ingest_docs
            result = ingest_docs()

            assert result == 0

    def test_ingest_docs_no_documents(self):
        """Test ingest_docs returns 0 when no markdown files found."""
        with patch("backend.agent.rag.ingest.DOCS_DIR") as mock_dir, \
             patch("llama_index.core.SimpleDirectoryReader") as mock_reader, \
             patch("llama_index.core.Settings"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.core.node_parser.SentenceSplitter"):

            mock_dir.exists.return_value = True
            mock_reader_instance = MagicMock()
            mock_reader_instance.load_data.return_value = []
            mock_reader.return_value = mock_reader_instance

            from backend.agent.rag.ingest import ingest_docs
            result = ingest_docs()

            assert result == 0

    def test_ingest_private_texts_no_file(self):
        """Test ingest_private_texts returns 0 when JSONL doesn't exist."""
        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_path, \
             patch("llama_index.core.Settings"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.core.node_parser.SentenceSplitter"):
            mock_path.exists.return_value = False

            from backend.agent.rag.ingest import ingest_private_texts
            result = ingest_private_texts()

            assert result == 0

    def test_ingest_private_texts_empty_file(self):
        """Test ingest_private_texts with empty JSONL file."""
        with patch("backend.agent.rag.ingest.JSONL_PATH") as mock_path, \
             patch("llama_index.core.Settings"), \
             patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding"), \
             patch("llama_index.core.node_parser.SentenceSplitter"), \
             patch("builtins.open", mock_open(read_data="")):

            mock_path.exists.return_value = True

            from backend.agent.rag.ingest import ingest_private_texts
            result = ingest_private_texts()

            assert result == 0


class TestSharedLlmClient:
    """Tests for backend.llm.client module (shared LLM infrastructure)."""

    def test_requires_max_completion_tokens_o1(self):
        """Test that o1 models require max_completion_tokens."""
        from backend.llm.client import _requires_max_completion_tokens

        assert _requires_max_completion_tokens("o1-preview") is True
        assert _requires_max_completion_tokens("o1-mini") is True
        assert _requires_max_completion_tokens("O1-preview") is True

    def test_requires_max_completion_tokens_gpt4(self):
        """Test that gpt-4 models don't require max_completion_tokens."""
        from backend.llm.client import _requires_max_completion_tokens

        assert _requires_max_completion_tokens("gpt-4o") is False
        assert _requires_max_completion_tokens("gpt-4o-mini") is False
        assert _requires_max_completion_tokens("gpt-4-turbo") is False

    def test_get_chat_model_no_api_key(self):
        """Test get_chat_model raises when no API key."""
        from backend.llm.client import get_chat_model

        # Clear cache
        get_chat_model.cache_clear()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_chat_model("gpt-4o-mini", 0.0, 1024)

    def test_get_chat_model_with_api_key(self):
        """Test get_chat_model creates ChatOpenAI with API key."""
        from backend.llm.client import get_chat_model

        # Clear cache
        get_chat_model.cache_clear()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch("backend.llm.client.ChatOpenAI") as mock_chat:

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
        from backend.llm.client import get_chat_model

        get_chat_model.cache_clear()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch("backend.llm.client.ChatOpenAI") as mock_chat:

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
        from backend.llm.client import call_llm, get_chat_model

        get_chat_model.cache_clear()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch("backend.llm.client.ChatOpenAI") as mock_chat:

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
        from backend.llm.client import call_llm, get_chat_model

        get_chat_model.cache_clear()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch("backend.llm.client.ChatOpenAI") as mock_chat:

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


class TestEvalShared:
    """Tests for backend.eval.shared module - LLM judge functions."""

    def test_run_llm_judge_success(self):
        """Test run_llm_judge with successful response."""
        from backend.eval.shared import run_llm_judge

        with patch("backend.eval.llm_client.call_llm") as mock_call:
            mock_call.return_value = '{"answer_relevance": 5, "explanation": "Good answer"}'

            result = run_llm_judge("test prompt", "system prompt")

            assert result["answer_relevance"] == 5
            assert result["explanation"] == "Good answer"

    def test_run_llm_judge_empty_response(self):
        """Test run_llm_judge with empty response."""
        from backend.eval.shared import run_llm_judge

        with patch("backend.eval.llm_client.call_llm") as mock_call:
            mock_call.return_value = ""

            result = run_llm_judge("test prompt", "system prompt")

            assert "error" in result
            assert "Empty response" in result["error"]

    def test_run_llm_judge_exception(self):
        """Test run_llm_judge handles exceptions."""
        from backend.eval.shared import run_llm_judge

        with patch("backend.eval.llm_client.call_llm") as mock_call:
            mock_call.side_effect = Exception("API error")

            result = run_llm_judge("test prompt", "system prompt")

            assert "error" in result
            assert "API error" in result["error"]

    def test_finalize_eval_cli_all_pass(self):
        """Test finalize_eval_cli when all checks pass."""
        from backend.eval.shared import finalize_eval_cli
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"

            slo_checks = [
                ("Accuracy", True, "95%", "90%"),
                ("Latency", True, "100ms", "200ms"),
            ]

            exit_code = finalize_eval_cli(
                primary_score=0.95,
                slo_checks=slo_checks,
                baseline_path=baseline_path,
                score_key="accuracy",
            )

            assert exit_code == 0

    def test_finalize_eval_cli_slo_failure(self):
        """Test finalize_eval_cli when SLO fails."""
        from backend.eval.shared import finalize_eval_cli
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"

            slo_checks = [
                ("Accuracy", False, "85%", "90%"),  # Failed
                ("Latency", True, "100ms", "200ms"),
            ]

            exit_code = finalize_eval_cli(
                primary_score=0.85,
                slo_checks=slo_checks,
                baseline_path=baseline_path,
                score_key="accuracy",
            )

            assert exit_code == 1

    def test_finalize_eval_cli_with_baseline_save(self):
        """Test finalize_eval_cli saves baseline when requested."""
        from backend.eval.shared import finalize_eval_cli
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"

            slo_checks = [("Accuracy", True, "95%", "90%")]

            finalize_eval_cli(
                primary_score=0.95,
                slo_checks=slo_checks,
                baseline_path=baseline_path,
                score_key="accuracy",
                set_baseline=True,
                baseline_data={"accuracy": 0.95},
            )

            assert baseline_path.exists()

    def test_save_eval_results(self):
        """Test save_eval_results writes JSON file."""
        from backend.eval.shared import save_eval_results
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")

            summary = MagicMock()
            summary.model_dump.return_value = {"accuracy": 0.95}

            results = [{"id": 1, "score": 0.9}, {"id": 2, "score": 1.0}]

            save_eval_results(
                output_path=output_path,
                summary=summary,
                results=results,
                result_mapper=lambda r: r,
            )

            assert os.path.exists(output_path)
            with open(output_path) as f:
                data = json.load(f)
            assert data["summary"]["accuracy"] == 0.95
            assert len(data["results"]) == 2


class TestEvalFormatting:
    """Tests for uncovered formatting functions."""

    def test_build_eval_table(self):
        """Test build_eval_table creates table with sections."""
        from backend.eval.formatting import build_eval_table

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
        assert len(table.columns) == 4
