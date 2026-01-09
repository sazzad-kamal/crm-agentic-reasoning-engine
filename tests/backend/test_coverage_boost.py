"""
Comprehensive test coverage boost for backend modules.

Targets modules with <98% coverage to bring them to target level.
This file works with the existing autouse mock_llm fixture from conftest.py.
"""

import json
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest


# =============================================================================
# Tests for backend/agent/fetch/handlers/common.py (82% -> 98%)
# Lines: 176-188, 200-207
# =============================================================================


class TestCommonPrivateTextsLoading:
    """Tests for _load_private_texts and _load_attachments functions."""

    def test_load_private_texts_file_exists(self):
        """Test loading private texts from JSONL file."""
        from backend.agent.fetch.handlers import common

        # Clear cache to ensure fresh load
        common._load_private_texts.cache_clear()

        jsonl_content = (
            '{"company_id": "COMP001", "text": "Note 1", "id": "n1"}\n'
            '{"company_id": "COMP001", "text": "Note 2", "id": "n2"}\n'
            '{"company_id": "COMP002", "text": "Note 3", "id": "n3"}\n'
            '\n'  # Empty line should be skipped
            '{"company_id": "", "text": "No company"}\n'  # Empty company_id should be skipped
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "private_texts.jsonl"
            jsonl_path.write_text(jsonl_content)

            with patch.object(common, "_get_csv_path", return_value=Path(tmpdir)):
                result = common._load_private_texts()

        assert "COMP001" in result
        assert len(result["COMP001"]) == 2
        assert "COMP002" in result
        assert len(result["COMP002"]) == 1
        # Empty company_id should not create an entry
        assert "" not in result

        # Clear cache after test
        common._load_private_texts.cache_clear()

    def test_load_private_texts_invalid_json(self):
        """Test loading private texts with invalid JSON line."""
        from backend.agent.fetch.handlers import common

        common._load_private_texts.cache_clear()

        jsonl_content = (
            '{"company_id": "COMP001", "text": "Valid"}\n'
            'not valid json\n'
            '{"company_id": "COMP002", "text": "Also valid"}\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "private_texts.jsonl"
            jsonl_path.write_text(jsonl_content)

            with patch.object(common, "_get_csv_path", return_value=Path(tmpdir)):
                result = common._load_private_texts()

        # Should have valid entries despite invalid JSON line
        assert "COMP001" in result
        assert "COMP002" in result

        common._load_private_texts.cache_clear()

    def test_load_attachments_file_exists(self):
        """Test loading attachments from CSV file."""
        from backend.agent.fetch.handlers import common

        common._load_attachments.cache_clear()

        csv_content = (
            "opportunity_id,name,type\n"
            "OPP001,doc1.pdf,proposal\n"
            "OPP001,doc2.pdf,contract\n"
            "OPP002,doc3.pdf,quote\n"
            ",noopportunity.pdf,other\n"  # Empty opportunity_id should be skipped
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "attachments.csv"
            csv_path.write_text(csv_content)

            with patch.object(common, "_get_csv_path", return_value=Path(tmpdir)):
                result = common._load_attachments()

        assert "OPP001" in result
        assert len(result["OPP001"]) == 2
        assert "OPP002" in result
        assert len(result["OPP002"]) == 1

        common._load_attachments.cache_clear()

    def test_load_attachments_file_not_exists(self):
        """Test loading attachments when CSV file doesn't exist."""
        from backend.agent.fetch.handlers import common

        common._load_attachments.cache_clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(common, "_get_csv_path", return_value=Path(tmpdir)):
                result = common._load_attachments()

        assert result == {}

        common._load_attachments.cache_clear()

    def test_load_private_texts_file_not_exists(self):
        """Test loading private texts when JSONL file doesn't exist."""
        from backend.agent.fetch.handlers import common

        common._load_private_texts.cache_clear()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(common, "_get_csv_path", return_value=Path(tmpdir)):
                result = common._load_private_texts()

        assert result == {}

        common._load_private_texts.cache_clear()


class TestLookupCompanySuccessPath:
    """Tests for lookup_company success path (lines 146-156)."""

    def test_lookup_company_success(self):
        """Test lookup_company when company is found."""
        from backend.agent.fetch.handlers.common import IntentResult, lookup_company, empty_raw_data

        result = IntentResult(raw_data=empty_raw_data())

        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = "COMP001"
        mock_ds.get_company.return_value = {
            "company_id": "COMP001",
            "name": "Test Company",
            "industry": "Tech"
        }
        mock_ds.get_contacts_for_company.return_value = [
            {"contact_id": "CON001", "name": "John Doe"}
        ]

        with patch("backend.agent.fetch.handlers.common.get_datastore", return_value=mock_ds):
            found = lookup_company(result, "Test Company")

        assert found is True
        assert result.company_data["found"] is True
        assert result.company_data["company"]["name"] == "Test Company"
        assert len(result.company_data["contacts"]) == 1
        assert len(result.sources) == 1
        assert result.sources[0].type == "company"
        assert result.resolved_company_id == "COMP001"
        assert result.raw_data["companies"] == [mock_ds.get_company.return_value]

    def test_lookup_company_resolved_but_not_found(self):
        """Test lookup_company when company resolves but doesn't exist."""
        from backend.agent.fetch.handlers.common import IntentResult, lookup_company, empty_raw_data

        result = IntentResult(raw_data=empty_raw_data())

        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = "COMP001"
        mock_ds.get_company.return_value = None  # Company doesn't exist

        with patch("backend.agent.fetch.handlers.common.get_datastore", return_value=mock_ds):
            found = lookup_company(result, "Test Company")

        assert found is False
        assert result.company_data["found"] is False


# =============================================================================
# Tests for backend/eval/langsmith.py
# =============================================================================


class TestLangsmithLatency:
    """Tests for LangSmith latency functions."""

    def test_get_latency_breakdown_no_api_key(self):
        """Test get_latency_breakdown when API key not set."""
        from backend.eval import langsmith

        with patch.dict("os.environ", {"LANGCHAIN_API_KEY": ""}, clear=False):
            with patch.object(langsmith, "os") as mock_os:
                mock_os.getenv.return_value = None
                result = langsmith.get_latency_breakdown()

        assert result == {}

    def test_get_latency_breakdown_with_runs(self):
        """Test get_latency_breakdown with mock LangSmith runs."""
        from backend.eval import langsmith
        from datetime import datetime, timedelta

        mock_runs = []
        base_time = datetime.utcnow()

        # Create mock runs with start_time and end_time
        for i, (name, duration_ms) in enumerate([
            ("route", 100), ("fetch_crm", 200), ("answer", 500), ("followup", 150)
        ]):
            mock_run = MagicMock()
            mock_run.name = name
            mock_run.start_time = base_time
            mock_run.end_time = base_time + timedelta(milliseconds=duration_ms)
            mock_runs.append(mock_run)

        mock_client_instance = MagicMock()
        mock_client_instance.list_runs.return_value = mock_runs

        mock_client_class = MagicMock(return_value=mock_client_instance)

        # Patch the langsmith module
        mock_langsmith_mod = MagicMock()
        mock_langsmith_mod.Client = mock_client_class

        with patch.dict("os.environ", {"LANGCHAIN_API_KEY": "test-key"}), \
             patch.dict(sys.modules, {"langsmith": mock_langsmith_mod}):
            result = langsmith.get_latency_breakdown()

        assert "route" in result
        assert result["route"]["count"] == 1
        assert result["route"]["avg_ms"] == 100.0

    def test_get_latency_breakdown_client_error(self):
        """Test get_latency_breakdown when client throws error."""
        from backend.eval import langsmith

        mock_client_instance = MagicMock()
        mock_client_instance.list_runs.side_effect = Exception("API error")

        mock_client_class = MagicMock(return_value=mock_client_instance)

        mock_langsmith_mod = MagicMock()
        mock_langsmith_mod.Client = mock_client_class

        with patch.dict("os.environ", {"LANGCHAIN_API_KEY": "test-key"}), \
             patch.dict(sys.modules, {"langsmith": mock_langsmith_mod}):
            result = langsmith.get_latency_breakdown()

        assert result == {}

    def test_get_latency_breakdown_no_runs(self):
        """Test get_latency_breakdown when no runs found."""
        from backend.eval import langsmith

        mock_client_instance = MagicMock()
        mock_client_instance.list_runs.return_value = []

        mock_client_class = MagicMock(return_value=mock_client_instance)

        mock_langsmith_mod = MagicMock()
        mock_langsmith_mod.Client = mock_client_class

        with patch.dict("os.environ", {"LANGCHAIN_API_KEY": "test-key"}), \
             patch.dict(sys.modules, {"langsmith": mock_langsmith_mod}):
            result = langsmith.get_latency_breakdown()

        assert result == {}

    def test_get_latency_percentages_empty(self):
        """Test get_latency_percentages with no breakdown."""
        from backend.eval import langsmith

        with patch.object(langsmith, "get_latency_breakdown", return_value={}):
            result = langsmith.get_latency_percentages()

        assert result == {}

    def test_get_latency_percentages_no_agent_nodes(self):
        """Test get_latency_percentages when no agent nodes found."""
        from backend.eval import langsmith

        # Return data with non-agent nodes
        with patch.object(langsmith, "get_latency_breakdown", return_value={
            "some_other_node": {"avg_ms": 100}
        }):
            result = langsmith.get_latency_percentages()

        assert result == {}

    def test_get_latency_percentages_success(self):
        """Test get_latency_percentages with valid data."""
        from backend.eval import langsmith

        breakdown = {
            "route": {"avg_ms": 100},
            "fetch_account": {"avg_ms": 200},
            "answer": {"avg_ms": 400},
            "followup": {"avg_ms": 300},
        }

        with patch.object(langsmith, "get_latency_breakdown", return_value=breakdown):
            result = langsmith.get_latency_percentages()

        assert "routing" in result
        assert "retrieval" in result
        assert "answer" in result
        assert "followup" in result
        # Total should be 1.0 (100%)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_get_latency_percentages_zero_total(self):
        """Test get_latency_percentages when all avg_ms are 0."""
        from backend.eval import langsmith

        breakdown = {
            "route": {"avg_ms": 0},
            "fetch_account": {"avg_ms": 0},
        }

        with patch.object(langsmith, "get_latency_breakdown", return_value=breakdown):
            result = langsmith.get_latency_percentages()

        assert result == {}


# =============================================================================
# Tests for backend/agent/rag/ingest.py (38% -> 98%)
# Lines: 50-114
# =============================================================================


class TestIngestPrivateTexts:
    """Tests for ingest_private_texts function."""

    def test_ingest_private_texts_empty_documents(self):
        """Test ingest returns 0 when JSONL has no valid documents."""
        from backend.agent.rag import ingest

        # Content with empty lines only
        jsonl_content = "\n\n\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            temp_path = Path(f.name)

        try:
            with patch.object(ingest, "JSONL_PATH", temp_path), \
                 patch.object(ingest, "close_qdrant_client"):
                # Mock llama_index imports
                mock_settings = MagicMock()
                mock_doc = MagicMock()
                mock_splitter = MagicMock()
                mock_embed = MagicMock()
                mock_storage = MagicMock()
                mock_index = MagicMock()
                mock_vector_store = MagicMock()

                with patch.dict(sys.modules, {
                    "llama_index.core": MagicMock(
                        Document=mock_doc,
                        Settings=mock_settings,
                        StorageContext=mock_storage,
                        VectorStoreIndex=mock_index,
                    ),
                    "llama_index.core.node_parser": MagicMock(SentenceSplitter=mock_splitter),
                    "llama_index.embeddings.huggingface": MagicMock(HuggingFaceEmbedding=mock_embed),
                    "llama_index.vector_stores.qdrant": MagicMock(QdrantVectorStore=mock_vector_store),
                }):
                    result = ingest.ingest_private_texts()

            assert result == 0
        finally:
            temp_path.unlink()

    def test_ingest_private_texts_with_documents(self):
        """Test ingest processes valid documents."""
        from backend.agent.rag import ingest

        jsonl_content = (
            '{"id": "doc1", "text": "Test content 1", "company_id": "COMP001", "type": "note", "title": "Note 1"}\n'
            '{"id": "doc2", "text": "Test content 2", "company_id": "COMP002", "type": "email", "title": "Email 1"}\n'
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            temp_path = Path(f.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            qdrant_path = Path(tmpdir) / "qdrant"

            try:
                # Create mock objects
                mock_document = MagicMock()
                mock_client = MagicMock()
                mock_client.collection_exists.return_value = True
                mock_vector_store = MagicMock()
                mock_storage_context = MagicMock()
                mock_index = MagicMock()

                with patch.object(ingest, "JSONL_PATH", temp_path), \
                     patch.object(ingest, "QDRANT_PATH", qdrant_path), \
                     patch.object(ingest, "close_qdrant_client"), \
                     patch.object(ingest, "QdrantClient", return_value=mock_client):

                    # Mock llama_index
                    mock_core = MagicMock()
                    mock_core.Document = lambda **kwargs: MagicMock(**kwargs)
                    mock_core.Settings = MagicMock()
                    mock_core.StorageContext.from_defaults.return_value = mock_storage_context
                    mock_core.VectorStoreIndex.from_documents.return_value = mock_index

                    mock_splitter = MagicMock()
                    mock_embed = MagicMock()
                    mock_qdrant_vs = MagicMock()
                    mock_qdrant_vs.QdrantVectorStore.return_value = mock_vector_store

                    with patch.dict(sys.modules, {
                        "llama_index.core": mock_core,
                        "llama_index.core.node_parser": MagicMock(SentenceSplitter=mock_splitter),
                        "llama_index.embeddings.huggingface": MagicMock(HuggingFaceEmbedding=mock_embed),
                        "llama_index.vector_stores.qdrant": mock_qdrant_vs,
                    }):
                        result = ingest.ingest_private_texts(recreate=True)

                # Should return document count
                assert result == 2
            finally:
                temp_path.unlink()

    def test_ingest_private_texts_json_decode_error(self):
        """Test ingest handles JSON decode errors gracefully."""
        from backend.agent.rag import ingest

        jsonl_content = (
            '{"id": "doc1", "text": "Valid", "company_id": "COMP001"}\n'
            'invalid json line\n'
            '{"id": "doc2", "text": "Also valid", "company_id": "COMP002"}\n'
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            temp_path = Path(f.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            qdrant_path = Path(tmpdir) / "qdrant"

            try:
                mock_client = MagicMock()
                mock_client.collection_exists.return_value = False
                mock_storage_context = MagicMock()

                with patch.object(ingest, "JSONL_PATH", temp_path), \
                     patch.object(ingest, "QDRANT_PATH", qdrant_path), \
                     patch.object(ingest, "close_qdrant_client"), \
                     patch.object(ingest, "QdrantClient", return_value=mock_client):

                    mock_core = MagicMock()
                    mock_core.Document = lambda **kwargs: MagicMock(**kwargs)
                    mock_core.Settings = MagicMock()
                    mock_core.StorageContext.from_defaults.return_value = mock_storage_context

                    with patch.dict(sys.modules, {
                        "llama_index.core": mock_core,
                        "llama_index.core.node_parser": MagicMock(),
                        "llama_index.embeddings.huggingface": MagicMock(),
                        "llama_index.vector_stores.qdrant": MagicMock(),
                    }):
                        result = ingest.ingest_private_texts(recreate=False)

                # Should still process valid documents (2)
                assert result == 2
            finally:
                temp_path.unlink()


# =============================================================================
# Tests for backend/eval/ragas_judge.py (34% -> 98%)
# Lines: 57-66, 77, 82, 106-177
# =============================================================================


class TestRagasJudge:
    """Tests for RAGAS evaluation functions."""

    def test_mock_evaluate_single_with_answer_and_context(self):
        """Test mock evaluation with answer and context."""
        from backend.eval.ragas_judge import _mock_evaluate_single

        result = _mock_evaluate_single(
            question="What is the pipeline?",
            answer="The pipeline has 5 deals worth $1M",
            contexts=["Pipeline data shows 5 deals"],
            reference_answer="5 deals in pipeline"
        )

        assert result["answer_relevancy"] == 0.85
        assert result["faithfulness"] == 0.80
        assert result["context_precision"] == 0.75
        assert result["context_recall"] == 0.70
        assert result["answer_correctness"] == 0.65

    def test_mock_evaluate_single_with_answer_no_context(self):
        """Test mock evaluation with answer but no context."""
        from backend.eval.ragas_judge import _mock_evaluate_single

        result = _mock_evaluate_single(
            question="What is the pipeline?",
            answer="The pipeline has 5 deals",
            contexts=["No context provided"],
        )

        assert result["answer_relevancy"] == 0.70
        assert result["faithfulness"] == 0.50
        assert result["context_precision"] == 0.0

    def test_mock_evaluate_single_no_answer(self):
        """Test mock evaluation with no answer."""
        from backend.eval.ragas_judge import _mock_evaluate_single

        result = _mock_evaluate_single(
            question="What is the pipeline?",
            answer="",
            contexts=[],
        )

        assert result["answer_relevancy"] == 0.0
        assert result["faithfulness"] == 0.0

    def test_evaluate_single_mock_mode(self):
        """Test evaluate_single in mock mode."""
        from backend.eval import ragas_judge

        with patch.object(ragas_judge, "_is_mock_mode", return_value=True):
            result = ragas_judge.evaluate_single(
                question="Test question",
                answer="Test answer with enough content",
                contexts=["Test context"],
            )

        assert "answer_relevancy" in result
        assert "faithfulness" in result

    def test_is_mock_mode_true(self):
        """Test _is_mock_mode returns True when MOCK_LLM=1."""
        from backend.eval.ragas_judge import _is_mock_mode

        with patch.dict("os.environ", {"MOCK_LLM": "1"}):
            assert _is_mock_mode() is True

    def test_is_mock_mode_false(self):
        """Test _is_mock_mode returns False when MOCK_LLM not set."""
        from backend.eval.ragas_judge import _is_mock_mode

        with patch.dict("os.environ", {}, clear=True):
            assert _is_mock_mode() is False


# =============================================================================
# Tests for backend/eval/models.py (99% -> 100%)
# Lines: 81, 87
# =============================================================================


class TestEvalModels:
    """Tests for E2EEvalResult.passed property edge cases."""

    def test_e2e_eval_result_passed_with_error(self):
        """Test E2EEvalResult.passed returns False when error exists."""
        from backend.eval.models import E2EEvalResult

        result = E2EEvalResult(
            test_case_id="test1",
            question="Test question",
            category="rag",
            answer="Test answer",
            answer_relevance=0.9,
            faithfulness=0.9,
            has_sources=True,
            latency_ms=100,
            total_tokens=50,
            error="Some error occurred"
        )

        assert result.passed is False

    def test_e2e_eval_result_passed_rag_high_scores(self):
        """Test E2EEvalResult.passed returns True for RAG with high scores."""
        from backend.eval.models import E2EEvalResult

        result = E2EEvalResult(
            test_case_id="test1",
            question="Test question",
            category="rag",
            answer="Test answer",
            answer_relevance=0.8,
            faithfulness=0.8,
            has_sources=True,
            latency_ms=100,
            total_tokens=50,
        )

        assert result.passed is True

    def test_e2e_eval_result_passed_rag_low_scores(self):
        """Test E2EEvalResult.passed returns False for RAG with low scores."""
        from backend.eval.models import E2EEvalResult

        result = E2EEvalResult(
            test_case_id="test1",
            question="Test question",
            category="rag",
            answer="Test answer",
            answer_relevance=0.5,  # Below 0.7 threshold
            faithfulness=0.8,
            has_sources=True,
            latency_ms=100,
            total_tokens=50,
        )

        assert result.passed is False

    def test_e2e_eval_result_security_refusal_correct(self):
        """Test E2EEvalResult.passed for security with correct refusal."""
        from backend.eval.models import E2EEvalResult

        result = E2EEvalResult(
            test_case_id="test1",
            question="Malicious prompt",
            category="adversarial",
            answer="I cannot help with that",
            answer_relevance=0.0,
            faithfulness=0.0,
            has_sources=False,
            latency_ms=100,
            total_tokens=50,
            expected_refusal=True,
            refusal_correct=True,
            has_forbidden_content=False,
        )

        assert result.passed is True

    def test_e2e_eval_result_security_forbidden_content(self):
        """Test E2EEvalResult.passed fails when forbidden content present."""
        from backend.eval.models import E2EEvalResult

        result = E2EEvalResult(
            test_case_id="test1",
            question="Malicious prompt",
            category="anti_hallucination",
            answer="Some forbidden response",
            answer_relevance=0.0,
            faithfulness=0.0,
            has_sources=False,
            latency_ms=100,
            total_tokens=50,
            expected_refusal=True,
            refusal_correct=True,
            has_forbidden_content=True,  # This should fail
        )

        assert result.passed is False


# =============================================================================
# Tests for answer/llm.py - build_answer_input (testable without mocks)
# =============================================================================


class TestBuildAnswerInput:
    """Tests for build_answer_input function."""

    def test_build_answer_input(self):
        """Test build_answer_input creates correct dict."""
        from backend.agent.answer.llm import build_answer_input

        result = build_answer_input(
            question="What is the status?",
            conversation_history_section="History...",
            company_section="Company...",
            activities_section="Activities...",
            history_section="Past...",
            pipeline_section="Pipeline...",
            renewals_section="Renewals...",
            account_context_section="Context...",
            contacts_section="Contacts...",
            groups_section="Groups...",
            attachments_section="Attachments...",
        )

        assert result["question"] == "What is the status?"
        assert result["conversation_history_section"] == "History..."
        assert result["account_context_section"] == "Context..."
        assert result["contacts_section"] == "Contacts..."
        assert result["groups_section"] == "Groups..."
        assert result["attachments_section"] == "Attachments..."

    def test_build_answer_input_defaults(self):
        """Test build_answer_input with default optional parameters."""
        from backend.agent.answer.llm import build_answer_input

        result = build_answer_input(
            question="Test question",
            conversation_history_section="",
            company_section="",
            activities_section="",
            history_section="",
            pipeline_section="",
            renewals_section="",
        )

        assert result["question"] == "Test question"
        assert result["account_context_section"] == ""
        assert result["contacts_section"] == ""
