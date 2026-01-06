"""
Integration tests for conversation history in the agent pipeline.

Tests that conversation history flows correctly through the agent using LangGraph checkpointer.

Run with: pytest tests/backend/agent/test_conversation_history.py -v
"""

import pytest

from backend.agent.core.state import AgentState, Message, format_history_for_prompt
from backend.agent.graph import clear_thread


class TestMessageType:
    """Tests for the Message TypedDict."""

    def test_message_structure(self):
        """Test that Message has the expected structure."""
        msg: Message = {
            "role": "user",
            "content": "Hello",
        }
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

    def test_message_roles(self):
        """Test both user and assistant roles."""
        user_msg: Message = {"role": "user", "content": "Hi"}
        assistant_msg: Message = {"role": "assistant", "content": "Hello!"}
        assert user_msg["role"] == "user"
        assert assistant_msg["role"] == "assistant"


class TestAgentStateWithMessages:
    """Tests for AgentState with messages field."""

    def test_state_with_empty_messages(self):
        """Test creating state with empty messages."""
        state: AgentState = {
            "question": "Hello",
            "messages": [],
        }
        assert state["messages"] == []

    def test_state_with_messages(self):
        """Test creating state with conversation history."""
        messages: list[Message] = [
            {"role": "user", "content": "Tell me about Acme"},
            {"role": "assistant", "content": "Acme is..."},
        ]
        state: AgentState = {
            "question": "What about their contacts?",
            "messages": messages,
        }
        assert len(state["messages"]) == 2

    def test_state_with_conversation_history(self):
        """Test that conversation_history field works."""
        state: AgentState = {
            "question": "Follow up",
            "messages": [],
            "conversation_history": "User: Hello\nAssistant: Hi there!",
        }
        assert "Hello" in state["conversation_history"]


class TestFormatHistoryForPrompt:
    """Tests for the format_history_for_prompt function."""

    def test_format_empty(self):
        """Test formatting empty history."""
        result = format_history_for_prompt([])
        assert result == ""

    def test_format_single_turn(self):
        """Test formatting a single turn."""
        messages: list[Message] = [
            {"role": "user", "content": "What is Acme's status?"},
        ]
        result = format_history_for_prompt(messages)
        assert "User: What is Acme's status?" in result

    def test_format_multi_turn(self):
        """Test formatting multiple turns."""
        messages: list[Message] = [
            {"role": "user", "content": "Tell me about Acme"},
            {"role": "assistant", "content": "Acme Manufacturing is a mid-market account"},
            {"role": "user", "content": "What about their contacts?"},
        ]
        result = format_history_for_prompt(messages)

        assert "User: Tell me about Acme" in result
        assert "Assistant: Acme Manufacturing is a mid-market account" in result
        assert "User: What about their contacts?" in result

    def test_format_truncates_long_content(self):
        """Test that long content is truncated."""
        long_content = "A" * 250
        messages: list[Message] = [
            {"role": "assistant", "content": long_content},
        ]
        result = format_history_for_prompt(messages)

        assert "..." in result
        # Should be truncated to ~200 chars + "..."
        assert "A" * 200 in result

    def test_format_respects_max_messages(self):
        """Test that only recent messages are included."""
        messages: list[Message] = [
            {"role": "user", "content": f"Question {i}"}
            for i in range(10)
        ]
        result = format_history_for_prompt(messages, max_messages=3)

        # Should only have the last 3
        assert "Question 7" in result
        assert "Question 8" in result
        assert "Question 9" in result
        assert "Question 0" not in result
        assert "Question 6" not in result


class TestClearThread:
    """Tests for clear_thread function."""

    def test_clear_nonexistent_session(self):
        """Test clearing a non-existent session is safe."""
        clear_thread("nonexistent")  # Should not raise

    def test_clear_none_session(self):
        """Test clearing None session is safe."""
        clear_thread(None)  # Should not raise
