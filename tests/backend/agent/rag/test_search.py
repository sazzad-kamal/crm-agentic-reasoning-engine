"""
Tests for RAG search module.

Covers search_entity_context function with mocked dependencies.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch


# Mock llama_index modules before any imports
mock_huggingface_module = MagicMock()
mock_core_module = MagicMock()
mock_qdrant_module = MagicMock()


class TestToolEntityRag:
    """Tests for search_entity_context function."""

    def _setup_llama_mocks(self, mock_retriever):
        """Setup common llama_index mocks."""
        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        mock_index_cls = MagicMock()
        mock_index_cls.from_vector_store.return_value = mock_index

        mock_settings = MagicMock()
        mock_vector_store_cls = MagicMock()
        mock_embed_cls = MagicMock()

        mock_core = MagicMock()
        mock_core.Settings = mock_settings
        mock_core.VectorStoreIndex = mock_index_cls

        # Mock the postprocessor module for SentenceTransformerRerank
        mock_postprocessor = MagicMock()
        mock_core_postprocessor = MagicMock()
        mock_core_postprocessor.SentenceTransformerRerank = MagicMock()

        mock_hf = MagicMock()
        mock_hf.HuggingFaceEmbedding = mock_embed_cls

        mock_qdrant_vs = MagicMock()
        mock_qdrant_vs.QdrantVectorStore = mock_vector_store_cls

        return {
            "llama_index.core": mock_core,
            "llama_index.core.postprocessor": mock_core_postprocessor,
            "llama_index.embeddings.huggingface": mock_hf,
            "llama_index.vector_stores.qdrant": mock_qdrant_vs,
        }

    @pytest.mark.no_mock_llm
    def test_search_entity_context_success(self):
        """Test successful entity RAG retrieval."""
        mock_node1 = MagicMock()
        mock_node1.text = "Meeting notes from last week."
        mock_node1.metadata = {"type": "note", "source_id": "note_001", "title": "Sales Call Notes"}

        mock_node2 = MagicMock()
        mock_node2.text = "Proposal document content."
        mock_node2.metadata = {"type": "attachment", "source_id": "att_001", "title": "Q4 Proposal"}

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node1, mock_node2]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            # Clear cached import
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            # Patch where it's used, not where it's defined
            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = search.search_entity_context("What were the meeting notes?", {"company_id": "COMP001"})

        assert "Meeting notes" in context
        assert "Proposal document" in context
        assert len(sources) == 2
        assert sources[0]["type"] == "note"
        assert sources[0]["id"] == "note_001"
        assert sources[1]["type"] == "attachment"

    @pytest.mark.no_mock_llm
    def test_search_entity_context_empty_results(self):
        """Test entity RAG with no results."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = search.search_entity_context("Unknown query", {"company_id": "COMP001"})

        assert context == ""
        assert sources == []

    @pytest.mark.no_mock_llm
    def test_search_entity_context_exception(self):
        """Test account RAG handles exceptions gracefully."""
        from backend.agent.fetch.rag import search

        # Reset globals and patch to raise exception during initialization
        search._vector_index = None
        search._embed_model = None
        search._reranker = None

        with patch.object(search, "get_qdrant_client", side_effect=Exception("Client error")):
            context, sources = search.search_entity_context("Any question", {"company_id": "COMP001"})

        assert context == ""
        assert sources == []

    @pytest.mark.no_mock_llm
    def test_search_entity_context_metadata_defaults(self):
        """Test account RAG uses defaults for missing metadata."""
        mock_node = MagicMock()
        mock_node.text = "Content with minimal metadata"
        mock_node.metadata = {}  # Empty metadata

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            # Import after setting up llama_index mocks
            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            # Patch where it's used, not where it's defined
            with patch.object(search, "get_qdrant_client") as mock_client_fn:
                mock_client_fn.return_value = MagicMock()
                context, sources = search.search_entity_context("Question", {"company_id": "COMP001"})

        assert len(sources) == 1
        assert sources[0]["type"] == "note"  # Default type
        assert sources[0]["id"] == "unknown"  # Default id

    @pytest.mark.no_mock_llm
    def test_search_entity_context_doc_id_fallback(self):
        """Test account RAG uses doc_id when source_id missing."""
        mock_node = MagicMock()
        mock_node.text = "Some content"
        mock_node.metadata = {"type": "attachment", "doc_id": "doc_123"}  # No source_id

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client_fn:
                mock_client_fn.return_value = MagicMock()
                context, sources = search.search_entity_context("Question", {"company_id": "COMP001"})

        assert len(sources) == 1
        assert sources[0]["id"] == "doc_123"

    @pytest.mark.no_mock_llm
    def test_search_entity_context_multi_entity_filter(self):
        """Test account RAG builds compound filter from multiple entity IDs."""
        mock_node = MagicMock()
        mock_node.text = "Lisa discussed pricing with the client."
        mock_node.metadata = {"type": "note", "source_id": "note_001", "title": "Call Notes"}

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                # Pass multiple entity IDs
                context, sources = search.search_entity_context(
                    "What did Lisa say?",
                    {"company_id": "COMP001", "contact_id": "CONT001", "opportunity_id": "OPP001"}
                )

        assert "Lisa discussed pricing" in context
        assert len(sources) == 1

    @pytest.mark.no_mock_llm
    def test_search_entity_context_company_type(self):
        """Test entity RAG returns company description content."""
        mock_node = MagicMock()
        mock_node.text = "Enterprise customer since 2020. CFO is key decision maker."
        mock_node.metadata = {
            "type": "company",
            "source_id": "company::ACME-MFG",
            "title": "Acme Manufacturing",
            "company_id": "ACME-MFG",
        }

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = search.search_entity_context(
                    "What should I know about Acme?",
                    {"company_id": "ACME-MFG"}
                )

        assert "Enterprise customer" in context
        assert sources[0]["type"] == "company"

    @pytest.mark.no_mock_llm
    def test_search_entity_context_contact_type(self):
        """Test entity RAG returns contact notes content."""
        mock_node = MagicMock()
        mock_node.text = "Prefers email over calls. Technical background. Champions our product."
        mock_node.metadata = {
            "type": "contact",
            "source_id": "contact::C-ACME-ANNA",
            "title": "Anna Lopez",
            "company_id": "ACME-MFG",
            "contact_id": "C-ACME-ANNA",
        }

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = search.search_entity_context(
                    "What's Anna's communication style?",
                    {"contact_id": "C-ACME-ANNA"}
                )

        assert "Prefers email" in context
        assert sources[0]["type"] == "contact"

    @pytest.mark.no_mock_llm
    def test_search_entity_context_opportunity_type(self):
        """Test entity RAG returns opportunity notes content."""
        mock_node = MagicMock()
        mock_node.text = "Deal is stalled due to pricing concerns. CFO wants ROI justification."
        mock_node.metadata = {
            "type": "opportunity",
            "source_id": "opp::OPP-ACME-UPGRADE",
            "title": "Pro to Enterprise Upgrade",
            "company_id": "ACME-MFG",
            "opportunity_id": "OPP-ACME-UPGRADE",
        }

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = search.search_entity_context(
                    "What's blocking the Acme upgrade?",
                    {"opportunity_id": "OPP-ACME-UPGRADE"}
                )

        assert "pricing concerns" in context
        assert sources[0]["type"] == "opportunity"

    @pytest.mark.no_mock_llm
    def test_search_entity_context_activity_type(self):
        """Test entity RAG returns activity description content."""
        mock_node = MagicMock()
        mock_node.text = "Reviewed Enterprise pricing with Anna. CFO approval needed before proceeding."
        mock_node.metadata = {
            "type": "activity",
            "source_id": "activity::ACT-ACME-1",
            "title": "Follow up on proposal",
            "company_id": "ACME-MFG",
            "contact_id": "C-ACME-ANNA",
        }

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = search.search_entity_context(
                    "What happened in the Acme follow-up?",
                    {"company_id": "ACME-MFG"}
                )

        assert "CFO approval" in context
        assert sources[0]["type"] == "activity"

    @pytest.mark.no_mock_llm
    def test_search_entity_context_history_type(self):
        """Test entity RAG returns history description content."""
        mock_node = MagicMock()
        mock_node.text = "QBR meeting with Lisa. Discussed expansion to marketing team."
        mock_node.metadata = {
            "type": "history",
            "source_id": "history::HIST-BETA-1",
            "title": "Quarterly Business Review",
            "company_id": "BETA-TECH",
        }

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = search.search_entity_context(
                    "What did we discuss in the Beta Tech QBR?",
                    {"company_id": "BETA-TECH"}
                )

        assert "expansion to marketing" in context
        assert sources[0]["type"] == "history"

    @pytest.mark.no_mock_llm
    def test_search_entity_context_attachment_type(self):
        """Test entity RAG returns attachment summary content."""
        mock_node = MagicMock()
        mock_node.text = "Proposal includes Enterprise pricing tier breakdown and ROI projections."
        mock_node.metadata = {
            "type": "attachment",
            "source_id": "attachment::ATT-ACME-PROPOSAL",
            "title": "Enterprise Proposal",
            "company_id": "ACME-MFG",
            "opportunity_id": "OPP-ACME-UPGRADE",
        }

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = search.search_entity_context(
                    "What's in the Acme proposal?",
                    {"opportunity_id": "OPP-ACME-UPGRADE"}
                )

        assert "pricing tier" in context
        assert sources[0]["type"] == "attachment"

    @pytest.mark.no_mock_llm
    def test_search_entity_context_mixed_entity_types(self):
        """Test entity RAG returns mixed content types for a company query."""
        mock_company = MagicMock()
        mock_company.text = "Enterprise customer since 2020."
        mock_company.metadata = {"type": "company", "source_id": "company::ACME", "title": "Acme"}

        mock_contact = MagicMock()
        mock_contact.text = "Anna champions the product internally."
        mock_contact.metadata = {"type": "contact", "source_id": "contact::ANNA", "title": "Anna Lopez"}

        mock_history = MagicMock()
        mock_history.text = "Last meeting discussed expansion plans."
        mock_history.metadata = {"type": "history", "source_id": "hist::1", "title": "Meeting"}

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_company, mock_contact, mock_history]

        llama_mocks = self._setup_llama_mocks(mock_retriever)

        with patch.dict(sys.modules, llama_mocks):
            if "backend.agent.fetch.rag.search" in sys.modules:
                del sys.modules["backend.agent.fetch.rag.search"]

            from backend.agent.fetch.rag import search

            # Reset the cached globals to force re-initialization
            search._vector_index = None
            search._embed_model = None
            search._reranker = None

            with patch.object(search, "get_qdrant_client") as mock_client:
                mock_client.return_value = MagicMock()
                context, sources = search.search_entity_context(
                    "What should I know about Acme before my call?",
                    {"company_id": "ACME"}
                )

        # Should have all three content types
        assert len(sources) == 3
        source_types = {s["type"] for s in sources}
        assert source_types == {"company", "contact", "history"}
        # Context should include all text
        assert "Enterprise customer" in context
        assert "champions" in context
        assert "expansion" in context
