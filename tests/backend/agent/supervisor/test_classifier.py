"""Tests for supervisor intent classifier."""

import pytest

from backend.agent.supervisor.classifier import Intent, classify_intent


class TestIntentEnum:
    """Tests for Intent enum."""

    def test_intent_values(self):
        """Test Intent enum has expected values."""
        assert Intent.DATA_QUERY.value == "data_query"
        assert Intent.COMPARE.value == "compare"
        assert Intent.TREND.value == "trend"
        assert Intent.COMPLEX.value == "complex"
        assert Intent.EXPORT.value == "export"
        assert Intent.HEALTH.value == "health"
        assert Intent.DOCS.value == "docs"
        assert Intent.CLARIFY.value == "clarify"
        assert Intent.HELP.value == "help"


class TestClassifyIntentHeuristics:
    """Tests for quick heuristic classification (no LLM call)."""

    # --- CLARIFY: Short/vague inputs ---

    def test_very_short_input_classified_as_clarify(self):
        """Very short inputs should trigger clarification."""
        assert classify_intent("yes") == Intent.CLARIFY
        assert classify_intent("no") == Intent.CLARIFY
        assert classify_intent("ok") == Intent.CLARIFY
        assert classify_intent("it") == Intent.CLARIFY

    def test_contextual_pronouns_classified_as_clarify(self):
        """Contextual pronouns without context need clarification."""
        assert classify_intent("that") == Intent.CLARIFY
        assert classify_intent("this") == Intent.CLARIFY

    # --- HELP: Help-seeking inputs ---

    def test_help_keyword_classified_as_help(self):
        """'help' keyword should classify as HELP."""
        assert classify_intent("help") == Intent.HELP
        assert classify_intent("Help me") == Intent.HELP

    def test_greeting_classified_as_help(self):
        """Greetings should classify as HELP."""
        assert classify_intent("hello") == Intent.HELP
        assert classify_intent("hi there") == Intent.HELP

    def test_capabilities_question_classified_as_help(self):
        """Questions about capabilities should classify as HELP."""
        assert classify_intent("what can you do") == Intent.HELP
        assert classify_intent("What can you do?") == Intent.HELP

    def test_how_to_question_classified_as_help(self):
        """How-to questions should classify as HELP."""
        assert classify_intent("how do i use this") == Intent.HELP
        assert classify_intent("how does this work") == Intent.HELP

    # --- DOCS: Documentation/how-to questions about Act! CRM ---

    def test_act_keyword_classified_as_docs(self):
        """Questions with Act! keyword should classify as DOCS."""
        assert classify_intent("How do I use Act! CRM?") == Intent.DOCS
        assert classify_intent("What features does Act! have?") == Intent.DOCS

    def test_how_to_with_product_action_classified_as_docs(self):
        """How-to questions with product actions should classify as DOCS."""
        assert classify_intent("How do I import contacts?") == Intent.DOCS
        assert classify_intent("How to create a group?") == Intent.DOCS

    def test_standalone_feature_not_docs(self):
        """Standalone feature names (without how-to) should NOT be DOCS."""
        # "opportunity stages" is a data query, not docs
        assert classify_intent("opportunity stages") == Intent.DATA_QUERY

    # --- DATA_QUERY: Data-seeking inputs ---

    def test_show_command_classified_as_data_query(self):
        """'Show' commands should classify as DATA_QUERY."""
        assert classify_intent("show me all companies") == Intent.DATA_QUERY
        assert classify_intent("Show pipeline") == Intent.DATA_QUERY

    def test_list_command_classified_as_data_query(self):
        """'List' commands should classify as DATA_QUERY."""
        assert classify_intent("list all contacts") == Intent.DATA_QUERY
        assert classify_intent("List deals closing this month") == Intent.DATA_QUERY

    def test_find_command_classified_as_data_query(self):
        """'Find' commands should classify as DATA_QUERY."""
        assert classify_intent("find Acme Corp") == Intent.DATA_QUERY
        assert classify_intent("Find recent activities") == Intent.DATA_QUERY

    def test_what_question_classified_as_data_query(self):
        """'What' questions about data should classify as DATA_QUERY."""
        assert classify_intent("what is the total revenue") == Intent.DATA_QUERY
        assert classify_intent("What deals are in negotiation?") == Intent.DATA_QUERY

    def test_who_question_classified_as_data_query(self):
        """'Who' questions should classify as DATA_QUERY."""
        assert classify_intent("who owns the Acme account") == Intent.DATA_QUERY

    def test_count_question_classified_as_data_query(self):
        """Count questions should classify as DATA_QUERY."""
        assert classify_intent("how many deals closed") == Intent.DATA_QUERY
        assert classify_intent("how much revenue this quarter") == Intent.DATA_QUERY

    def test_crm_terms_classified_as_data_query(self):
        """Questions with CRM terms should classify as DATA_QUERY."""
        assert classify_intent("pipeline summary") == Intent.DATA_QUERY
        assert classify_intent("renewal dates") == Intent.DATA_QUERY
        assert classify_intent("opportunity stages") == Intent.DATA_QUERY
        assert classify_intent("contact activities") == Intent.DATA_QUERY

    # --- COMPARE: Comparison queries ---

    def test_vs_comparison_classified_as_compare(self):
        """'X vs Y' queries should classify as COMPARE."""
        assert classify_intent("Compare Q1 vs Q2 revenue") == Intent.COMPARE
        assert classify_intent("Sales this year vs last year") == Intent.COMPARE

    def test_versus_comparison_classified_as_compare(self):
        """'X versus Y' queries should classify as COMPARE."""
        assert classify_intent("Revenue this quarter versus last") == Intent.COMPARE

    def test_compare_keyword_classified_as_compare(self):
        """'compare' keyword should classify as COMPARE."""
        assert classify_intent("Compare Acme and TechCorp deals") == Intent.COMPARE

    # --- TREND: Time-series queries ---

    def test_trend_keyword_classified_as_trend(self):
        """'trend' keyword should classify as TREND."""
        assert classify_intent("Show revenue trend") == Intent.TREND
        assert classify_intent("What's the pipeline trend?") == Intent.TREND

    def test_over_time_classified_as_trend(self):
        """'over time' queries should classify as TREND."""
        assert classify_intent("How has revenue changed over time") == Intent.TREND

    def test_growth_classified_as_trend(self):
        """'growth' queries should classify as TREND."""
        assert classify_intent("Show pipeline growth") == Intent.TREND

    def test_monthly_quarterly_classified_as_trend(self):
        """Monthly/quarterly queries should classify as TREND."""
        assert classify_intent("Monthly revenue breakdown") == Intent.TREND
        assert classify_intent("Quarterly sales performance") == Intent.TREND

    # --- EXPORT: Export queries ---

    def test_export_keyword_classified_as_export(self):
        """'export' keyword should classify as EXPORT."""
        assert classify_intent("Export deals to CSV") == Intent.EXPORT
        assert classify_intent("Export all contacts") == Intent.EXPORT

    def test_download_classified_as_export(self):
        """'download' keyword should classify as EXPORT."""
        assert classify_intent("Download the data") == Intent.EXPORT

    def test_csv_pdf_classified_as_export(self):
        """File format keywords should classify as EXPORT."""
        assert classify_intent("Get deals as CSV") == Intent.EXPORT
        assert classify_intent("Generate PDF report") == Intent.EXPORT

    # --- HEALTH: Health score queries ---

    def test_health_score_classified_as_health(self):
        """'health score' queries should classify as HEALTH."""
        assert classify_intent("What's Acme's health score?") == Intent.HEALTH
        assert classify_intent("Show account health") == Intent.HEALTH

    def test_at_risk_classified_as_health(self):
        """'at risk' queries should classify as HEALTH."""
        assert classify_intent("Which accounts are at risk?") == Intent.HEALTH

    # --- COMPLEX: Multi-part queries ---

    def test_multipart_with_and_classified_as_complex(self):
        """Multi-part queries with 'and' should classify as COMPLEX."""
        assert classify_intent("Show deals and compare Q1 vs Q2") == Intent.COMPLEX
        assert classify_intent("List companies and show their trends") == Intent.COMPLEX


class TestClassifyIntentWithHistory:
    """Tests for classification with conversation history."""

    def test_history_is_passed_to_classifier(self):
        """Verify history parameter is accepted."""
        # This should not raise
        result = classify_intent(
            "what about them",
            conversation_history="User: Show me Acme\nAssistant: Here are the details..."
        )
        # With context, this might be DATA_QUERY, but heuristics may still say CLARIFY
        assert result in (Intent.DATA_QUERY, Intent.CLARIFY)


class TestSupervisorNode:
    """Tests for the supervisor node function."""

    def test_supervisor_node_returns_intent_in_state(self):
        """Supervisor node should return intent and loop_count."""
        from backend.agent.supervisor.node import supervisor_node

        state = {"question": "show me all companies"}
        result = supervisor_node(state)

        assert "intent" in result
        assert result["intent"] == "data_query"
        assert "loop_count" in result
        assert result["loop_count"] == 0

    def test_supervisor_node_with_help_intent(self):
        """Supervisor node should classify help questions."""
        from backend.agent.supervisor.node import supervisor_node

        state = {"question": "help"}
        result = supervisor_node(state)

        assert result["intent"] == "help"

    def test_supervisor_node_with_clarify_intent(self):
        """Supervisor node should classify vague questions."""
        from backend.agent.supervisor.node import supervisor_node

        state = {"question": "yes"}
        result = supervisor_node(state)

        assert result["intent"] == "clarify"
