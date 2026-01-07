"""
Tests for the LangGraph agent structure with parallel fetch nodes.

Verifies:
- Graph has 6 nodes (route + 3 parallel fetch + answer + followup)
- All router intents are explicitly mapped
- Account RAG triggers for correct intents
"""

import pytest


class TestGraphStructure:
    """Tests for the parallel fetch graph structure."""

    def test_graph_has_six_nodes(self):
        """Verify the graph has 6 nodes with parallel fetch."""
        from backend.agent.graph import agent_graph

        # Get node names from the graph
        node_names = set(agent_graph.nodes.keys())

        # Should have exactly these 6 nodes (plus __start__ and __end__)
        # 3 parallel fetch nodes: fetch_crm, fetch_docs, fetch_account
        expected_nodes = {"route", "fetch_crm", "fetch_docs", "fetch_account", "answer", "followup"}

        # Filter out internal nodes
        actual_nodes = {n for n in node_names if not n.startswith("__")}

        assert actual_nodes == expected_nodes, (
            f"Expected 6 nodes {expected_nodes}, got {actual_nodes}"
        )

    def test_old_nodes_removed(self):
        """Verify old nodes module was removed and consolidated into vertical slices."""
        import backend.agent as agent_module

        # The old nodes module should NOT exist anymore
        assert not hasattr(agent_module, "nodes"), "nodes module should be removed"

        # The new structure should exist
        assert hasattr(agent_module, "route"), "route module should exist"
        assert hasattr(agent_module, "fetch"), "fetch module should exist"
        assert hasattr(agent_module, "answer"), "answer module should exist"
        assert hasattr(agent_module, "followup"), "followup module should exist"

    def test_nodes_modules_exist(self):
        """Verify nodes submodules export the correct functions."""
        from backend.agent.route.node import route_node
        from backend.agent.fetch.fetch_crm import fetch_crm_node
        from backend.agent.fetch.fetch_docs import fetch_docs_node
        from backend.agent.fetch.fetch_account import fetch_account_node, ACCOUNT_RAG_INTENTS
        from backend.agent.answer.node import answer_node
        from backend.agent.followup.node import followup_node

        # Verify all functions are callable
        assert callable(route_node)
        assert callable(fetch_crm_node)
        assert callable(fetch_docs_node)
        assert callable(fetch_account_node)
        assert callable(answer_node)
        assert callable(followup_node)
        assert isinstance(ACCOUNT_RAG_INTENTS, frozenset)


class TestIntentHandlers:
    """Tests for explicit intent handler mappings."""

    def test_all_router_intents_mapped(self):
        """Verify all possible router intents have explicit handlers."""
        from backend.agent.fetch.handlers import INTENT_HANDLERS

        # All intents that the router can return (from llm_router.py)
        router_intents = {
            "company_status",
            "renewals",
            "pipeline",
            "activities",
            "history",
            "account_context",
            "general",
        }

        # All intents should be in INTENT_HANDLERS
        for intent in router_intents:
            assert intent in INTENT_HANDLERS, (
                f"Intent '{intent}' not explicitly mapped in INTENT_HANDLERS"
            )

    def test_account_context_intent_mapped(self):
        """Verify account_context intent is explicitly mapped."""
        from backend.agent.fetch.handlers import INTENT_HANDLERS

        assert "account_context" in INTENT_HANDLERS
        # It should map to handle_company_status
        handler = INTENT_HANDLERS["account_context"]
        assert handler.__name__ == "handle_company_status"

    def test_history_intent_mapped(self):
        """Verify history intent is explicitly mapped (was implicit fallthrough)."""
        from backend.agent.fetch.handlers import INTENT_HANDLERS

        assert "history" in INTENT_HANDLERS
        handler = INTENT_HANDLERS["history"]
        assert handler.__name__ == "handle_company_status"

    def test_general_intent_mapped(self):
        """Verify general intent is explicitly mapped."""
        from backend.agent.fetch.handlers import INTENT_HANDLERS

        assert "general" in INTENT_HANDLERS


class TestAccountRAGTrigger:
    """Tests for Account RAG trigger conditions."""

    def test_account_rag_trigger_intents(self):
        """Verify Account RAG triggers for the correct intents using the constant."""
        from backend.agent.fetch.fetch_account import ACCOUNT_RAG_INTENTS

        # Expected intents that should trigger Account RAG
        expected_intents = {"account_context", "company_status", "history", "pipeline"}

        assert ACCOUNT_RAG_INTENTS == expected_intents, (
            f"ACCOUNT_RAG_INTENTS should be {expected_intents}, got {ACCOUNT_RAG_INTENTS}"
        )

    def test_account_rag_not_trigger_for_aggregate_intents(self):
        """Verify Account RAG does NOT trigger for aggregate intents."""
        from backend.agent.fetch.fetch_account import ACCOUNT_RAG_INTENTS

        # Aggregate intents that don't need Account RAG
        aggregate_intents = {"renewals", "pipeline_summary", "activities", "groups"}

        for intent in aggregate_intents:
            if intent != "pipeline":  # pipeline is special - it can be company-specific
                assert intent not in ACCOUNT_RAG_INTENTS, (
                    f"Aggregate intent '{intent}' should not be in ACCOUNT_RAG_INTENTS"
                )


class TestVerticalSliceStructure:
    """Tests for the vertical slice LLM structure."""

    def test_shared_llm_client_exists(self):
        """Verify shared LLM client exists at backend.llm."""
        from backend.llm.client import create_chain, get_chat_model, call_llm

        assert callable(create_chain)
        assert callable(get_chat_model)
        assert callable(call_llm)

    def test_answer_llm_module_exists(self):
        """Verify answer/llm.py exports the correct functions."""
        from backend.agent.answer.llm import (
            call_answer_chain,
            call_not_found_chain,
            get_answer_chain,
            build_answer_input,
            stream_answer_chain,
        )

        assert callable(call_answer_chain)
        assert callable(call_not_found_chain)
        assert callable(get_answer_chain)
        assert callable(build_answer_input)

    def test_followup_llm_module_exists(self):
        """Verify followup/llm.py exports the correct functions."""
        from backend.agent.followup.llm import (
            generate_follow_up_suggestions,
            FollowUpSuggestions,
        )

        assert callable(generate_follow_up_suggestions)
        assert FollowUpSuggestions is not None

    def test_fetch_rag_module_exists(self):
        """Verify fetch/rag.py exports the correct functions."""
        from backend.agent.fetch.rag import (
            call_docs_rag,
            call_account_rag,
        )

        assert callable(call_docs_rag)
        assert callable(call_account_rag)

    def test_agent_llm_folder_removed(self):
        """Verify agent/llm/ folder has been removed."""
        import importlib.util

        # These should NOT exist - find_spec raises ModuleNotFoundError when
        # parent exists but child doesn't
        try:
            spec = importlib.util.find_spec("backend.agent.llm.helpers")
            assert spec is None, "backend.agent.llm.helpers should not exist"
        except ModuleNotFoundError:
            pass  # Expected - module doesn't exist
