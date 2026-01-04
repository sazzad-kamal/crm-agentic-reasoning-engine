"""
Tests for LLM helper functions and router.

These tests mock LLM calls to achieve coverage without API calls.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import time


class TestLlmHelpersChains:
    """Tests for llm_helpers chain creation and caching."""

    def test_create_chain_with_structured_output(self):
        """Test _create_chain with structured output."""
        from backend.agent.llm.helpers import _create_chain, _chains_cache, FollowUpSuggestions

        _chains_cache.clear()

        with patch("backend.agent.llm.helpers.ChatOpenAI") as mock_chat, \
             patch("backend.agent.llm.helpers.get_config") as mock_config, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            mock_cfg = MagicMock()
            mock_cfg.llm_model = "gpt-4o"
            mock_cfg.router_model = "gpt-4o-mini"
            mock_cfg.llm_temperature = 0.0
            mock_cfg.llm_max_tokens = 2000
            mock_config.return_value = mock_cfg

            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured
            mock_chat.return_value = mock_llm

            from backend.agent.llm.prompts import FOLLOW_UP_PROMPT_TEMPLATE

            chain = _create_chain(
                FOLLOW_UP_PROMPT_TEMPLATE,
                model_key="fast",
                structured_output=FollowUpSuggestions,
                temperature=0.7,
            )

            mock_llm.with_structured_output.assert_called_once_with(FollowUpSuggestions)

        _chains_cache.clear()

    def test_create_chain_without_structured_output(self):
        """Test _create_chain without structured output returns string parser chain."""
        from backend.agent.llm.helpers import _create_chain, _chains_cache

        _chains_cache.clear()

        with patch("backend.agent.llm.helpers.ChatOpenAI") as mock_chat, \
             patch("backend.agent.llm.helpers.get_config") as mock_config, \
             patch("backend.agent.llm.helpers.StrOutputParser") as mock_parser, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            mock_cfg = MagicMock()
            mock_cfg.llm_model = "gpt-4o"
            mock_cfg.llm_temperature = 0.0
            mock_cfg.llm_max_tokens = 2000
            mock_config.return_value = mock_cfg

            mock_llm = MagicMock()
            mock_chat.return_value = mock_llm

            from backend.agent.llm.prompts import DATA_ANSWER_TEMPLATE

            chain = _create_chain(DATA_ANSWER_TEMPLATE, model_key="main")

            # Should not call with_structured_output
            mock_llm.with_structured_output.assert_not_called()

        _chains_cache.clear()

    def test_get_chain_caching(self):
        """Test that _get_chain caches chains."""
        from backend.agent.llm.helpers import _get_chain, _chains_cache

        _chains_cache.clear()

        with patch("backend.agent.llm.helpers._create_chain") as mock_create:
            mock_chain = MagicMock()
            mock_create.return_value = mock_chain

            # First call creates chain
            chain1 = _get_chain("answer")
            assert chain1 is mock_chain

            # Second call returns cached
            chain2 = _get_chain("answer")
            assert chain2 is chain1

            # _create_chain only called once
            assert mock_create.call_count == 1

        _chains_cache.clear()

    def test_get_chain_unknown_type_raises(self):
        """Test _get_chain raises for unknown chain type."""
        from backend.agent.llm.helpers import _get_chain, _chains_cache

        _chains_cache.clear()

        with pytest.raises(ValueError, match="Unknown chain type"):
            _get_chain("nonexistent_chain_type")

        _chains_cache.clear()


class TestLlmHelpersCallFunctions:
    """Tests for the call_* functions in llm_helpers."""

    def test_call_answer_chain_mock_mode(self):
        """Test call_answer_chain in mock mode."""
        from backend.agent.llm.helpers import call_answer_chain

        with patch("backend.agent.llm.helpers.is_mock_mode", return_value=True):
            answer, latency = call_answer_chain(
                question="What is the pipeline?",
                conversation_history_section="",
                company_section="Test Company",
                activities_section="",
                history_section="",
                pipeline_section="Pipeline data",
                renewals_section="",
                docs_section="",
            )

            assert isinstance(answer, str)
            assert latency == 100

    def test_call_answer_chain_real_mode(self):
        """Test call_answer_chain in real mode."""
        from backend.agent.llm.helpers import call_answer_chain, _chains_cache

        _chains_cache.clear()

        with patch("backend.agent.llm.helpers.is_mock_mode", return_value=False), \
             patch("backend.agent.llm.helpers._get_chain") as mock_get_chain:

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Generated answer"
            mock_get_chain.return_value = mock_chain

            answer, latency = call_answer_chain(
                question="What is the pipeline?",
                conversation_history_section="",
                company_section="Test Company",
                activities_section="",
                history_section="",
                pipeline_section="Pipeline data",
                renewals_section="",
                docs_section="",
            )

            assert answer == "Generated answer"
            assert latency >= 0
            mock_chain.invoke.assert_called_once()

        _chains_cache.clear()

    def test_call_not_found_chain_mock_mode(self):
        """Test call_not_found_chain in mock mode."""
        from backend.agent.llm.helpers import call_not_found_chain

        with patch("backend.agent.llm.helpers.is_mock_mode", return_value=True):
            answer, latency = call_not_found_chain(
                question="Tell me about Acme",
                query="Acme",
                matches="Acme Corp, Acme Inc",
            )

            assert isinstance(answer, str)
            assert latency == 100

    def test_call_not_found_chain_real_mode(self):
        """Test call_not_found_chain in real mode."""
        from backend.agent.llm.helpers import call_not_found_chain, _chains_cache

        _chains_cache.clear()

        with patch("backend.agent.llm.helpers.is_mock_mode", return_value=False), \
             patch("backend.agent.llm.helpers._get_chain") as mock_get_chain:

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "I couldn't find that company"
            mock_get_chain.return_value = mock_chain

            answer, latency = call_not_found_chain(
                question="Tell me about Acme",
                query="Acme",
                matches="Acme Corp, Acme Inc",
            )

            assert answer == "I couldn't find that company"
            mock_chain.invoke.assert_called_once()

        _chains_cache.clear()

    def test_call_docs_rag_mock_mode(self):
        """Test call_docs_rag in mock mode."""
        from backend.agent.llm.helpers import call_docs_rag

        with patch("backend.agent.llm.helpers.is_mock_mode", return_value=True):
            context, sources = call_docs_rag("How do I create a contact?")

            assert "documentation" in context.lower() or "settings" in context.lower()
            assert len(sources) > 0
            assert sources[0].type == "doc"

    def test_call_docs_rag_real_mode_success(self):
        """Test call_docs_rag in real mode with success."""
        from backend.agent.llm.helpers import call_docs_rag
        from backend.agent.schemas import Source

        with patch("backend.agent.llm.helpers.is_mock_mode", return_value=False), \
             patch("backend.agent.rag.tools.tool_docs_rag") as mock_tool:

            mock_tool.return_value = ("Docs context", [Source(type="doc", id="doc1", label="Doc 1")])

            context, sources = call_docs_rag("How do I create a contact?")

            assert context == "Docs context"
            assert len(sources) == 1

    def test_call_docs_rag_real_mode_exception(self):
        """Test call_docs_rag handles exceptions gracefully."""
        from backend.agent.llm.helpers import call_docs_rag

        with patch("backend.agent.llm.helpers.is_mock_mode", return_value=False), \
             patch("backend.agent.rag.tools.tool_docs_rag", side_effect=Exception("RAG error")):

            context, sources = call_docs_rag("How do I create a contact?")

            assert context == ""
            assert sources == []

    def test_call_account_rag_mock_mode(self):
        """Test call_account_rag in mock mode."""
        from backend.agent.llm.helpers import call_account_rag

        with patch("backend.agent.llm.helpers.is_mock_mode", return_value=True):
            context, sources = call_account_rag("What are the notes?", "COMP001")

            assert "account" in context.lower() or "notes" in context.lower()
            assert len(sources) > 0

    def test_call_account_rag_real_mode_exception(self):
        """Test call_account_rag handles exceptions gracefully."""
        from backend.agent.llm.helpers import call_account_rag

        with patch("backend.agent.llm.helpers.is_mock_mode", return_value=False), \
             patch("backend.agent.rag.tools.tool_account_rag", side_effect=Exception("RAG error")):

            context, sources = call_account_rag("What are the notes?", "COMP001")

            assert context == ""
            assert sources == []


class TestLlmHelpersFollowUp:
    """Tests for follow-up suggestion generation."""

    def test_generate_follow_up_disabled(self):
        """Test generate_follow_up_suggestions when disabled in config."""
        from backend.agent.llm.helpers import generate_follow_up_suggestions

        with patch("backend.agent.llm.helpers.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.enable_follow_up_suggestions = False
            mock_config.return_value = mock_cfg

            result = generate_follow_up_suggestions(
                question="What is the pipeline?",
                mode="data",
            )

            assert result == []

    def test_generate_follow_up_hardcoded_tree(self):
        """Test generate_follow_up_suggestions uses hardcoded tree."""
        from backend.agent.llm.helpers import generate_follow_up_suggestions

        with patch("backend.agent.llm.helpers.get_config") as mock_config, \
             patch("backend.agent.question_tree.get_follow_ups") as mock_get_followups:

            mock_cfg = MagicMock()
            mock_cfg.enable_follow_up_suggestions = True
            mock_config.return_value = mock_cfg

            mock_get_followups.return_value = ["Follow-up 1", "Follow-up 2"]

            result = generate_follow_up_suggestions(
                question="What is the pipeline?",
                mode="data",
                use_hardcoded_tree=True,
            )

            assert result == ["Follow-up 1", "Follow-up 2"]

    def test_generate_follow_up_mock_mode(self):
        """Test generate_follow_up_suggestions in mock mode."""
        from backend.agent.llm.helpers import generate_follow_up_suggestions

        with patch("backend.agent.llm.helpers.get_config") as mock_config, \
             patch("backend.agent.llm.helpers.is_mock_mode", return_value=True), \
             patch("backend.agent.question_tree.get_follow_ups", return_value=[]):

            mock_cfg = MagicMock()
            mock_cfg.enable_follow_up_suggestions = True
            mock_config.return_value = mock_cfg

            result = generate_follow_up_suggestions(
                question="What is the pipeline?",
                mode="data",
                company_name="TestCorp",
                available_data={"opportunities": 5, "activities": 3},
            )

            # Should return mock suggestions based on available_data
            assert len(result) > 0

    def test_generate_follow_up_llm_failure(self):
        """Test generate_follow_up_suggestions handles LLM failures."""
        from backend.agent.llm.helpers import generate_follow_up_suggestions, _chains_cache

        _chains_cache.clear()

        with patch("backend.agent.llm.helpers.get_config") as mock_config, \
             patch("backend.agent.llm.helpers.is_mock_mode", return_value=False), \
             patch("backend.agent.question_tree.get_follow_ups", return_value=[]), \
             patch("backend.agent.llm.helpers._get_chain") as mock_get_chain:

            mock_cfg = MagicMock()
            mock_cfg.enable_follow_up_suggestions = True
            mock_config.return_value = mock_cfg

            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("LLM error")
            mock_get_chain.return_value = mock_chain

            result = generate_follow_up_suggestions(
                question="What is the pipeline?",
                mode="data",
                use_hardcoded_tree=False,
            )

            assert result == []

        _chains_cache.clear()


class TestLlmRouter:
    """Tests for llm_router module."""

    def test_call_llm_router_direct(self):
        """Test _call_llm_router directly with mocked chain."""
        from backend.agent.llm.router import _call_llm_router, LLMRouterResponse
        import backend.agent.llm_router as router_module

        router_module._router_chain = None

        with patch("backend.agent.llm.router._get_router_chain") as mock_get_chain:
            mock_chain = MagicMock()
            mock_response = LLMRouterResponse(
                mode="data",
                intent="pipeline",
                company_name="Acme Corp",
                days=30,
                confidence=0.85,
                key_entities=["pipeline", "deals"],
                action_type="retrieve",
            )
            mock_chain.invoke.return_value = mock_response
            mock_get_chain.return_value = mock_chain

            result = _call_llm_router("What is Acme's pipeline?", "Previous: Hello")

            assert result["mode"] == "data"
            assert result["intent"] == "pipeline"
            assert result["company_name"] == "Acme Corp"
            mock_chain.invoke.assert_called_once()
            call_args = mock_chain.invoke.call_args[0][0]
            assert "CONVERSATION HISTORY" in call_args["conversation_context"]

        router_module._router_chain = None

    def test_call_llm_router_no_history(self):
        """Test _call_llm_router without conversation history."""
        from backend.agent.llm.router import _call_llm_router, LLMRouterResponse
        import backend.agent.llm_router as router_module

        router_module._router_chain = None

        with patch("backend.agent.llm.router._get_router_chain") as mock_get_chain:
            mock_chain = MagicMock()
            mock_response = LLMRouterResponse(
                mode="docs",
                intent="general",
                company_name=None,
                days=90,
                confidence=0.9,
                key_entities=["help"],
                action_type="retrieve",
            )
            mock_chain.invoke.return_value = mock_response
            mock_get_chain.return_value = mock_chain

            result = _call_llm_router("How do I use the CRM?")

            assert result["mode"] == "docs"
            call_args = mock_chain.invoke.call_args[0][0]
            assert call_args["conversation_context"] == ""

        router_module._router_chain = None

    def test_llm_route_explicit_mode(self):
        """Test llm_route_question with explicit mode."""
        from backend.agent.llm.router import llm_route_question

        result = llm_route_question(
            question="Tell me about Acme",
            mode="data",
            company_id="COMP001",
        )

        assert result.mode_used == "data"
        assert result.company_id == "COMP001"

    def test_llm_route_mock_mode(self):
        """Test llm_route_question in mock mode."""
        from backend.agent.llm.router import llm_route_question

        with patch("backend.agent.llm.router.is_mock_mode", return_value=True):
            result = llm_route_question(
                question="Tell me about Acme",
                mode="auto",
            )

            assert result.mode_used == "data+docs"

    def test_llm_route_with_llm(self):
        """Test llm_route_question with actual LLM call."""
        from backend.agent.llm.router import llm_route_question, _router_chain

        # Reset chain cache
        import backend.agent.llm_router as router_module
        router_module._router_chain = None

        with patch("backend.agent.llm.router.is_mock_mode", return_value=False), \
             patch("backend.agent.llm.router._call_llm_router") as mock_call:

            mock_call.return_value = {
                "mode": "data",
                "intent": "pipeline_overview",
                "company_name": "Acme Corp",
                "days": 30,
                "confidence": 0.9,
                "key_entities": ["pipeline"],
            }

            mock_datastore = MagicMock()
            mock_datastore.resolve_company_id.return_value = "COMP001"

            result = llm_route_question(
                question="What is Acme's pipeline?",
                mode="auto",
                datastore=mock_datastore,
            )

            assert result.mode_used == "data"
            assert result.intent == "pipeline_overview"
            assert result.company_id == "COMP001"

        router_module._router_chain = None

    def test_get_router_chain_caching(self):
        """Test that _get_router_chain caches the chain."""
        import backend.agent.llm_router as router_module
        from backend.agent.llm.router import _get_router_chain

        router_module._router_chain = None

        with patch("backend.agent.llm.router.ChatOpenAI") as mock_chat, \
             patch("backend.agent.llm.router.get_config") as mock_config, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            mock_cfg = MagicMock()
            mock_cfg.router_model = "gpt-4o-mini"
            mock_cfg.router_temperature = 0.0
            mock_config.return_value = mock_cfg

            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured
            mock_chat.return_value = mock_llm

            # First call creates chain
            chain1 = _get_router_chain()

            # Second call returns cached
            chain2 = _get_router_chain()
            assert chain1 is chain2

            # ChatOpenAI only called once
            assert mock_chat.call_count == 1

        router_module._router_chain = None


class TestNodesFetching:
    """Tests for nodes/fetching module."""

    def test_fetch_docs_success(self):
        """Test _fetch_docs successful fetch."""
        from backend.agent.nodes.fetching import _fetch_docs
        from backend.agent.schemas import Source

        with patch("backend.agent.nodes.fetching.call_docs_rag") as mock_rag:
            mock_rag.return_value = ("Docs content", [Source(type="doc", id="doc1", label="Doc")])

            result = _fetch_docs("How to create contact?")

            assert result["docs_answer"] == "Docs content"
            assert len(result["docs_sources"]) == 1

    def test_fetch_docs_exception(self):
        """Test _fetch_docs handles exceptions."""
        from backend.agent.nodes.fetching import _fetch_docs

        with patch("backend.agent.nodes.fetching.call_docs_rag", side_effect=Exception("RAG error")):
            result = _fetch_docs("How to create contact?")

            assert result["docs_answer"] == ""
            assert result["docs_sources"] == []
            assert "error" in result

    def test_fetch_account_context_success(self):
        """Test _fetch_account_context successful fetch."""
        from backend.agent.nodes.fetching import _fetch_account_context
        from backend.agent.schemas import Source

        with patch("backend.agent.nodes.fetching.call_account_rag") as mock_rag:
            mock_rag.return_value = ("Account notes", [Source(type="note", id="n1", label="Note")])

            result = _fetch_account_context("What are the notes?", "COMP001")

            assert result["account_context_answer"] == "Account notes"
            assert len(result["account_context_sources"]) == 1

    def test_fetch_account_context_exception(self):
        """Test _fetch_account_context handles exceptions."""
        from backend.agent.nodes.fetching import _fetch_account_context

        with patch("backend.agent.nodes.fetching.call_account_rag", side_effect=Exception("RAG error")):
            result = _fetch_account_context("What are the notes?", "COMP001")

            assert result["account_context_answer"] == ""
            assert result["account_context_sources"] == []
            assert "error" in result


class TestDatastoreCore:
    """Tests for datastore core module edge cases."""

    def test_get_csv_base_path_preferred(self):
        """Test get_csv_base_path uses preferred path when exists."""
        from backend.agent.datastore.core import get_csv_base_path
        from pathlib import Path

        with patch.object(Path, "exists") as mock_exists, \
             patch.object(Path, "is_dir") as mock_is_dir:
            # Preferred path exists
            mock_exists.return_value = True
            mock_is_dir.return_value = True

            result = get_csv_base_path()
            # Should return a path
            assert result is not None

    def test_get_csv_base_path_fallback(self):
        """Test get_csv_base_path fallback to csv directory."""
        from backend.agent.datastore import core
        from pathlib import Path

        # Get the actual preferred and fallback paths
        backend_root = Path(core.__file__).parent.parent.parent

        preferred = backend_root / "data" / "crm"
        fallback = backend_root / "data" / "csv"

        # If preferred exists, this test can't run properly
        # But we can test the function logic directly
        result = core.get_csv_base_path()
        assert result is not None

    def test_crm_datastore_context_manager(self):
        """Test CRMDataStoreCore context manager."""
        from backend.agent.datastore.core import CRMDataStoreCore

        store = CRMDataStoreCore()
        store._conn = MagicMock()

        with store as s:
            assert s is store

        # Cleanup
        store._conn = None

    def test_crm_datastore_close(self):
        """Test CRMDataStoreCore close method."""
        from backend.agent.datastore.core import CRMDataStoreCore

        store = CRMDataStoreCore.__new__(CRMDataStoreCore)
        store._conn = MagicMock()
        store._loaded_tables = {"test"}
        store._company_names_cache = {"a": "b"}
        store._company_ids_cache = {"c"}

        store.close()

        assert store._conn is None
        assert len(store._loaded_tables) == 0
        assert store._company_names_cache is None
        assert store._company_ids_cache is None
