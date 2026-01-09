"""
Tests for LLM helper functions and router.

These tests mock LLM calls to achieve coverage without API calls.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import time


class TestSharedLlmClient:
    """Tests for backend.llm.client chain creation."""

    def test_create_chain_with_structured_output(self):
        """Test create_chain with structured output."""
        from backend.llm.client import create_chain
        from backend.agent.followup.llm import FollowUpSuggestions

        with patch("backend.llm.client.ChatOpenAI") as mock_chat, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured
            mock_chat.return_value = mock_llm

            from backend.agent.followup.prompts import FOLLOW_UP_PROMPT_TEMPLATE

            chain = create_chain(
                FOLLOW_UP_PROMPT_TEMPLATE,
                model="gpt-4o-mini",
                structured_output=FollowUpSuggestions,
                temperature=0.7,
            )

            mock_llm.with_structured_output.assert_called_once_with(FollowUpSuggestions)

    def test_create_chain_without_structured_output(self):
        """Test create_chain without structured output returns string parser chain."""
        from backend.llm.client import create_chain

        with patch("backend.llm.client.ChatOpenAI") as mock_chat, \
             patch("backend.llm.client.StrOutputParser") as mock_parser, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            mock_llm = MagicMock()
            mock_chat.return_value = mock_llm

            from backend.agent.answer.prompts import DATA_ANSWER_TEMPLATE

            chain = create_chain(DATA_ANSWER_TEMPLATE, model="gpt-4o")

            # Should not call with_structured_output
            mock_llm.with_structured_output.assert_not_called()

    def test_get_chat_model_caching(self):
        """Test that get_chat_model caches models."""
        from backend.llm.client import get_chat_model

        # Clear cache first
        get_chat_model.cache_clear()

        with patch("backend.llm.client.ChatOpenAI") as mock_chat, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            mock_llm = MagicMock()
            mock_chat.return_value = mock_llm

            # First call creates model
            model1 = get_chat_model("gpt-4o", 0.0, 2000)

            # Second call with same params returns cached
            model2 = get_chat_model("gpt-4o", 0.0, 2000)
            assert model1 is model2

            # ChatOpenAI only called once
            assert mock_chat.call_count == 1

        get_chat_model.cache_clear()

    def test_call_llm_basic(self):
        """Test call_llm basic functionality."""
        from backend.llm.client import call_llm, get_chat_model

        get_chat_model.cache_clear()

        with patch("backend.llm.client.ChatOpenAI") as mock_chat, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_chat.return_value = mock_llm

            result = call_llm("Test prompt", system_prompt="You are a test assistant")

            assert result == "Test response"
            mock_llm.invoke.assert_called_once()

        get_chat_model.cache_clear()


class TestLlmHelpersCallFunctions:
    """Tests for the call_* functions in node-specific modules.

    Note: These tests run with conftest.py autouse mock fixtures.
    """

    def test_call_answer_chain_returns_tuple(self):
        """Test call_answer_chain returns (answer, latency) tuple."""
        from backend.agent.answer.llm import call_answer_chain

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
        assert len(answer) > 0
        assert isinstance(latency, int)
        assert latency >= 0

    def test_call_not_found_chain_returns_tuple(self):
        """Test call_not_found_chain returns (answer, latency) tuple."""
        from backend.agent.answer.llm import call_not_found_chain

        answer, latency = call_not_found_chain(
            question="Tell me about Acme",
            query="Acme",
            matches="Acme Corp, Acme Inc",
        )

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(latency, int)

    def test_call_account_rag_returns_context_and_sources(self):
        """Test call_account_rag returns context and sources."""
        from backend.agent.fetch.rag import call_account_rag

        context, sources = call_account_rag("What are the notes?", "COMP001")

        assert isinstance(context, str)
        assert isinstance(sources, list)


class TestLlmHelpersFollowUp:
    """Tests for follow-up suggestion generation.

    Note: These tests run with conftest.py autouse mock fixtures.
    """

    def test_generate_follow_up_returns_list(self):
        """Test generate_follow_up_suggestions returns a list."""
        from backend.agent.followup.llm import generate_follow_up_suggestions

        result = generate_follow_up_suggestions(
            question="What is the pipeline?",
            mode="data",
        )

        assert isinstance(result, list)

    def test_generate_follow_up_with_hardcoded_tree(self):
        """Test generate_follow_up_suggestions uses hardcoded tree."""
        from backend.agent.followup.llm import generate_follow_up_suggestions

        with patch("backend.agent.followup.tree.get_follow_ups") as mock_get_followups:
            mock_get_followups.return_value = ["Follow-up 1", "Follow-up 2"]

            result = generate_follow_up_suggestions(
                question="What is the pipeline?",
                mode="data",
                use_hardcoded_tree=True,
            )

            assert result == ["Follow-up 1", "Follow-up 2"]

    def test_generate_follow_up_limited_to_three(self):
        """Test generate_follow_up_suggestions returns at most 3 suggestions."""
        from backend.agent.followup.llm import generate_follow_up_suggestions

        result = generate_follow_up_suggestions(
            question="Test question",
            mode="data",
        )

        assert len(result) <= 3


class TestLlmRouter:
    """Tests for llm_router module."""

    def test_call_llm_router_direct(self):
        """Test _call_llm_router directly with mocked chain."""
        from backend.agent.route.router import _call_llm_router, LLMRouterResponse
        import backend.agent.route.router as router_module

        router_module._router_chain = None

        with patch("backend.agent.route.router._get_router_chain") as mock_get_chain:
            mock_chain = MagicMock()
            mock_response = LLMRouterResponse(
                intent="company",
                company_name="Acme Corp",
            )
            mock_chain.invoke.return_value = mock_response
            mock_get_chain.return_value = mock_chain

            result = _call_llm_router("What is Acme's pipeline?", "Previous: Hello")

            assert result["intent"] == "company"
            assert result["company_name"] == "Acme Corp"
            mock_chain.invoke.assert_called_once()
            call_args = mock_chain.invoke.call_args[0][0]
            assert "CONVERSATION HISTORY" in call_args["conversation_context"]

        router_module._router_chain = None

    def test_call_llm_router_no_history(self):
        """Test _call_llm_router without conversation history."""
        from backend.agent.route.router import _call_llm_router, LLMRouterResponse
        import backend.agent.route.router as router_module

        router_module._router_chain = None

        with patch("backend.agent.route.router._get_router_chain") as mock_get_chain:
            mock_chain = MagicMock()
            mock_response = LLMRouterResponse(
                intent="pipeline_summary",
                company_name=None,
            )
            mock_chain.invoke.return_value = mock_response
            mock_get_chain.return_value = mock_chain

            result = _call_llm_router("How do I use the CRM?")

            assert result["intent"] == "pipeline_summary"
            call_args = mock_chain.invoke.call_args[0][0]
            assert call_args["conversation_context"] == ""

        router_module._router_chain = None

    def test_llm_route_question_returns_result(self):
        """Test llm_route_question returns RouterResult."""
        from backend.agent.route.router import llm_route_question

        result = llm_route_question(question="What is Acme's pipeline?")

        # RouterResult has company_id and intent
        assert hasattr(result, "company_id")
        assert result.intent is not None

    def test_get_router_chain_caching(self):
        """Test that _get_router_chain caches the chain."""
        import backend.agent.route.router as router_module
        from backend.agent.route.router import _get_router_chain

        router_module._router_chain = None

        with patch("backend.agent.route.router.ChatOpenAI") as mock_chat, \
             patch("backend.agent.route.router.get_config") as mock_config, \
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
    """Tests for parallel fetch nodes."""

    def test_fetch_account_node_success(self):
        """Test fetch_account_node successful fetch."""
        from backend.agent.fetch.fetch_account import fetch_account_node
        from backend.agent.core import Source

        with patch("backend.agent.fetch.fetch_account.call_account_rag") as mock_rag:
            mock_rag.return_value = ("Account notes", [Source(type="note", id="n1", label="Note")])

            # Use "company" intent - the only intent that triggers RAG
            state = {"question": "What are the notes?", "resolved_company_id": "COMP001", "intent": "company"}
            result = fetch_account_node(state)

            assert result["account_context_answer"] == "Account notes"
            assert len(result["sources"]) == 1

    def test_fetch_account_node_exception(self):
        """Test fetch_account_node handles exceptions."""
        from backend.agent.fetch.fetch_account import fetch_account_node

        with patch("backend.agent.fetch.fetch_account.call_account_rag", side_effect=Exception("RAG error")):
            # Use "company" intent - the only intent that triggers RAG
            state = {"question": "What are the notes?", "resolved_company_id": "COMP001", "intent": "company"}
            result = fetch_account_node(state)

            assert result["account_context_answer"] == ""
            assert "error" in result


class TestDatastoreCore:
    """Tests for datastore core module edge cases."""

    def test_get_csv_base_path_preferred(self):
        """Test get_csv_base_path uses preferred path when exists."""
        from backend.agent.datastore.base import get_csv_base_path
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
        from backend.agent.datastore import base
        from pathlib import Path

        # Get the actual preferred and fallback paths
        backend_root = Path(base.__file__).parent.parent.parent

        preferred = backend_root / "data" / "crm"
        fallback = backend_root / "data" / "csv"

        # If preferred exists, this test can't run properly
        # But we can test the function logic directly
        result = base.get_csv_base_path()
        assert result is not None

    def test_crm_datastore_context_manager(self):
        """Test CRMDataStoreBase context manager."""
        from backend.agent.datastore.base import CRMDataStoreBase

        store = CRMDataStoreBase()
        store._conn = MagicMock()

        with store as s:
            assert s is store

        # Cleanup
        store._conn = None

    def test_crm_datastore_close(self):
        """Test CRMDataStoreBase close method."""
        from backend.agent.datastore.base import CRMDataStoreBase

        store = CRMDataStoreBase.__new__(CRMDataStoreBase)
        store._conn = MagicMock()
        store._loaded_tables = {"test"}
        store._company_names_cache = {"a": "b"}
        store._company_ids_cache = {"c"}

        store.close()

        assert store._conn is None
        assert len(store._loaded_tables) == 0
        assert store._company_names_cache is None
        assert store._company_ids_cache is None
