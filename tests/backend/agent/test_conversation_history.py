"""
Integration tests for conversation history in the agent pipeline.

Tests that conversation history flows correctly through the agent.

Run with: pytest backend/agent/tests/test_conversation_history.py -v
"""

import pytest

from backend.agent.nodes.state import AgentState, Message
from backend.agent.nodes.support.memory import (
    clear_session,
    _memory_store,
)
from backend.agent.nodes.support.formatters import format_conversation_history_section


@pytest.fixture(autouse=True)
def clean_memory():
    """Clear memory before and after each test."""
    _memory_store.clear()
    yield
    _memory_store.clear()


class TestMessageType:
    """Tests for the Message TypedDict."""

    def test_message_structure(self):
        """Test that Message has the expected structure."""
        msg: Message = {
            "role": "user",
            "content": "Hello",
            "company_id": "ACME-MFG",
        }
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"
        assert msg["company_id"] == "ACME-MFG"

    def test_message_optional_company(self):
        """Test that company_id can be None."""
        msg: Message = {
            "role": "assistant",
            "content": "Hi there",
            "company_id": None,
        }
        assert msg["company_id"] is None


class TestAgentStateWithMessages:
    """Tests for AgentState with messages field."""

    def test_state_with_empty_messages(self):
        """Test creating state with empty messages."""
        state: AgentState = {
            "question": "Hello",
            "mode": "auto",
            "messages": [],
        }
        assert state["messages"] == []

    def test_state_with_messages(self):
        """Test creating state with conversation history."""
        messages: list[Message] = [
            {"role": "user", "content": "Tell me about Acme", "company_id": None},
            {"role": "assistant", "content": "Acme is...", "company_id": "ACME-MFG"},
        ]
        state: AgentState = {
            "question": "What about their contacts?",
            "mode": "auto",
            "session_id": "test_session",
            "messages": messages,
        }
        assert len(state["messages"]) == 2
        assert state["messages"][1]["company_id"] == "ACME-MFG"


class TestFormatConversationHistorySection:
    """Tests for the conversation history formatter."""

    def test_format_empty(self):
        """Test formatting empty history."""
        result = format_conversation_history_section(None)
        assert result == ""

        result = format_conversation_history_section([])
        assert result == ""

    def test_format_single_turn(self):
        """Test formatting a single turn."""
        messages = [
            {"role": "user", "content": "What is Acme's status?", "company_id": None},
        ]
        result = format_conversation_history_section(messages)

        assert "=== RECENT CONVERSATION ===" in result
        assert "User: What is Acme's status?" in result

    def test_format_multi_turn(self):
        """Test formatting multiple turns."""
        messages = [
            {"role": "user", "content": "Tell me about Acme", "company_id": None},
            {"role": "assistant", "content": "Acme Manufacturing is a mid-market account", "company_id": "ACME-MFG"},
            {"role": "user", "content": "What about their contacts?", "company_id": None},
        ]
        result = format_conversation_history_section(messages)

        assert "User: Tell me about Acme" in result
        assert "Assistant: Acme Manufacturing is a mid-market account" in result
        assert "User: What about their contacts?" in result

    def test_format_truncates_long_content(self):
        """Test that long content is truncated."""
        long_content = "A" * 200
        messages = [
            {"role": "assistant", "content": long_content, "company_id": None},
        ]
        result = format_conversation_history_section(messages)

        assert "..." in result
        # Should be truncated to ~150 chars + "..."
        assert "A" * 150 in result

    def test_format_respects_max_messages(self):
        """Test that only recent messages are included."""
        messages = [
            {"role": "user", "content": f"Question {i}", "company_id": None}
            for i in range(10)
        ]
        result = format_conversation_history_section(messages, max_messages=3)

        # Should only have the last 3
        assert "Question 7" in result
        assert "Question 8" in result
        assert "Question 9" in result
        assert "Question 0" not in result
        assert "Question 6" not in result


class TestClearSession:
    """Tests for clear_session function."""

    def test_clear_session(self):
        """Test clearing a session removes stored data."""
        session_id = "test_clear"
        _memory_store[session_id] = [{"role": "user", "content": "test", "company_id": None}]
        
        clear_session(session_id)
        
        assert session_id not in _memory_store

    def test_clear_nonexistent_session(self):
        """Test clearing a non-existent session is safe."""
        clear_session("nonexistent")  # Should not raise

    def test_clear_none_session(self):
        """Test clearing None session is safe."""
        clear_session(None)  # Should not raise

