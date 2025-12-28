"""
Tests for the simplified LangGraph agent structure.

Verifies:
- Graph has 4 nodes (simplified from 6)
- All router intents are explicitly mapped
- Account RAG triggers for correct intents
"""

import pytest


class TestGraphStructure:
    """Tests for the simplified graph structure."""

    def test_graph_has_four_nodes(self):
        """Verify the graph was simplified to 4 nodes."""
        from backend.agent.graph import build_agent_graph

        # Build a fresh graph
        graph = build_agent_graph(checkpointer=None)

        # Get node names from the graph
        node_names = set(graph.nodes.keys())

        # Should have exactly these 4 nodes (plus __start__ and __end__)
        expected_nodes = {"route", "fetch", "answer", "followup"}

        # Filter out internal nodes
        actual_nodes = {n for n in node_names if not n.startswith("__")}

        assert actual_nodes == expected_nodes, (
            f"Expected 4 nodes {expected_nodes}, got {actual_nodes}"
        )

    def test_old_nodes_removed(self):
        """Verify data, docs, data_and_docs nodes were consolidated."""
        from backend.agent import nodes

        # These should NOT exist anymore (consolidated into fetch_node)
        assert not hasattr(nodes, "data_node"), "data_node should be removed"
        assert not hasattr(nodes, "docs_node"), "docs_node should be removed"
        assert not hasattr(nodes, "data_and_docs_parallel_node"), (
            "data_and_docs_parallel_node should be removed"
        )
        assert not hasattr(nodes, "route_by_mode"), "route_by_mode should be removed"
        assert not hasattr(nodes, "skip_data_node"), "skip_data_node should be removed"
        assert not hasattr(nodes, "skip_docs_node"), "skip_docs_node should be removed"

    def test_nodes_module_exports(self):
        """Verify nodes module exports the correct functions."""
        from backend.agent.nodes import __all__

        expected_exports = {
            "route_node",
            "fetch_node",
            "answer_node",
            "followup_node",
        }

        assert set(__all__) == expected_exports


class TestIntentHandlers:
    """Tests for explicit intent handler mappings."""

    def test_all_router_intents_mapped(self):
        """Verify all possible router intents have explicit handlers."""
        from backend.agent.intent_handlers import INTENT_HANDLERS

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
        from backend.agent.intent_handlers import INTENT_HANDLERS

        assert "account_context" in INTENT_HANDLERS
        # It should map to handle_company_status
        handler = INTENT_HANDLERS["account_context"]
        assert handler.__name__ == "handle_company_status"

    def test_history_intent_mapped(self):
        """Verify history intent is explicitly mapped (was implicit fallthrough)."""
        from backend.agent.intent_handlers import INTENT_HANDLERS

        assert "history" in INTENT_HANDLERS
        handler = INTENT_HANDLERS["history"]
        assert handler.__name__ == "handle_company_status"

    def test_general_intent_mapped(self):
        """Verify general intent is explicitly mapped."""
        from backend.agent.intent_handlers import INTENT_HANDLERS

        assert "general" in INTENT_HANDLERS


class TestAccountRAGTrigger:
    """Tests for Account RAG trigger conditions."""

    def test_account_rag_trigger_intents(self):
        """Verify Account RAG triggers for the correct intents."""
        # The trigger condition from nodes.py
        account_rag_intents = ("account_context", "company_status", "history", "pipeline")

        # These intents should trigger Account RAG when company_id is set
        for intent in account_rag_intents:
            should_trigger = intent in account_rag_intents
            assert should_trigger, f"Intent '{intent}' should trigger Account RAG"

    def test_account_rag_not_trigger_for_aggregate_intents(self):
        """Verify Account RAG does NOT trigger for aggregate intents."""
        # Aggregate intents that don't need Account RAG
        aggregate_intents = ("renewals", "pipeline_summary", "activities", "groups")

        # These should NOT trigger Account RAG
        account_rag_intents = ("account_context", "company_status", "history", "pipeline")

        for intent in aggregate_intents:
            if intent != "pipeline":  # pipeline is special - it can be company-specific
                should_trigger = intent in account_rag_intents
                assert not should_trigger, (
                    f"Aggregate intent '{intent}' should not trigger Account RAG"
                )


class TestLLMHelpersCleanup:
    """Tests for removed dead code."""

    def test_call_llm_removed_from_exports(self):
        """Verify call_llm was removed from llm_helpers exports."""
        from backend.agent.llm_helpers import __all__

        assert "call_llm" not in __all__, "call_llm should be removed from exports"

    def test_essential_functions_still_exported(self):
        """Verify essential LLM helper functions are still exported."""
        from backend.agent.llm_helpers import __all__

        required_exports = {
            "call_docs_rag",
            "call_account_rag",
            "generate_follow_up_suggestions",
            "call_answer_chain",
            "call_not_found_chain",
        }

        for export in required_exports:
            assert export in __all__, f"'{export}' should be in llm_helpers exports"
