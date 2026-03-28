"""Tests for answer node with intent handling and loop detection."""

from unittest.mock import MagicMock, patch

import pytest


class TestAnswerNodeIntents:
    """Tests for answer node intent handling."""

    def test_clarify_intent_generates_clarification(self):
        """Clarify intent should generate a clarification request."""
        from backend.agent.answer.node import answer_node

        state = {"question": "that one", "intent": "clarify", "loop_count": 0}
        result = answer_node(state)

        assert "answer" in result
        assert "more details" in result["answer"].lower() or "clarify" in result["answer"].lower()
        assert result["needs_more_data"] is False

    def test_help_intent_generates_help_response(self):
        """Help intent should generate a help/capabilities response."""
        from backend.agent.answer.node import answer_node

        state = {"question": "help", "intent": "help", "loop_count": 0}
        result = answer_node(state)

        assert "answer" in result
        assert "crm" in result["answer"].lower() or "data" in result["answer"].lower()
        assert result["needs_more_data"] is False

    @patch("backend.agent.answer.node.call_answer_chain")
    def test_data_query_intent_calls_answer_chain(self, mock_chain):
        """Data query intent should call the answer chain."""
        from backend.agent.answer.node import answer_node

        answer_text = "The revenue is $100k [E1].\n\nEvidence:\n- E1: revenue=100000"
        mock_chain.return_value = answer_text

        state = {
            "question": "what is the revenue",
            "intent": "data_query",
            "loop_count": 0,
            "sql_results": {"data": [{"revenue": 100000}]},
        }
        result = answer_node(state)

        mock_chain.assert_called_once()
        assert result["answer"] == answer_text


class TestAnswerNodeLoopDetection:
    """Tests for needs_more_data detection."""

    @patch("backend.agent.answer.node.call_answer_chain")
    def test_detects_data_not_available(self, mock_chain):
        """Answer with 'data not available' should trigger needs_more_data."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = "Answer: Data not available for company names.\n\nEvidence:\n- E1: N/A"

        state = {
            "question": "show company names",
            "intent": "data_query",
            "loop_count": 0,
            "sql_results": {"data": [{"id": 1}]},
        }
        result = answer_node(state)
        assert result["needs_more_data"] is True

    @patch("backend.agent.answer.node.call_answer_chain")
    def test_detects_no_data_found(self, mock_chain):
        """Answer with 'no data found' should trigger needs_more_data."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = "No data found for this query."

        state = {
            "question": "show contacts",
            "intent": "data_query",
            "loop_count": 0,
            "sql_results": {},
        }
        result = answer_node(state)
        assert result["needs_more_data"] is True

    @patch("backend.agent.answer.node.call_answer_chain")
    def test_respects_max_loop_count(self, mock_chain):
        """Should not request more data after max loops."""
        from backend.agent.answer.node import answer_node, MAX_LOOP_COUNT

        mock_chain.return_value = "Data not available."

        state = {
            "question": "show data",
            "intent": "data_query",
            "loop_count": MAX_LOOP_COUNT,
            "sql_results": {},
        }
        result = answer_node(state)
        assert result["needs_more_data"] is False

    @patch("backend.agent.answer.node.call_answer_chain")
    def test_increments_loop_count(self, mock_chain):
        """Loop count should be incremented after each answer."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = "Here is the answer [E1]."

        state = {
            "question": "show data",
            "intent": "data_query",
            "loop_count": 0,
            "sql_results": {"data": [{"id": 1}]},
        }
        result = answer_node(state)
        assert result["loop_count"] == 1

    @patch("backend.agent.answer.node.call_answer_chain")
    def test_unfetchable_data_does_not_trigger_loop(self, mock_chain):
        """'Question not answerable' should not trigger a loop."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = "Data not available (question not answerable from provided CRM DATA)."

        state = {
            "question": "what is the meaning of life",
            "intent": "data_query",
            "loop_count": 0,
            "sql_results": {},
        }
        result = answer_node(state)
        assert result["needs_more_data"] is False

    @patch("backend.agent.answer.node.call_answer_chain")
    def test_complete_answer_does_not_trigger_loop(self, mock_chain):
        """Complete answers should not trigger needs_more_data."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = "The total revenue is $500,000 [E1].\n\nEvidence:\n- E1: SUM(revenue)=500000"

        state = {
            "question": "total revenue",
            "intent": "data_query",
            "loop_count": 0,
            "sql_results": {"data": [{"total": 500000}]},
        }
        result = answer_node(state)
        assert result["needs_more_data"] is False


class TestAnswerNodeMessages:
    """Tests for message handling in answer node."""

    def test_adds_messages_for_clarify(self):
        """Clarify intent should add human and AI messages."""
        from backend.agent.answer.node import answer_node

        state = {"question": "yes", "intent": "clarify", "loop_count": 0}
        result = answer_node(state)

        assert "messages" in result
        assert len(result["messages"]) == 2

    @patch("backend.agent.answer.node.call_answer_chain")
    def test_adds_messages_for_data_query(self, mock_chain):
        """Data query should add human and AI messages."""
        from backend.agent.answer.node import answer_node

        mock_chain.return_value = "Answer here."

        state = {
            "question": "show data",
            "intent": "data_query",
            "loop_count": 0,
            "sql_results": {},
        }
        result = answer_node(state)

        assert "messages" in result
        assert len(result["messages"]) == 2
