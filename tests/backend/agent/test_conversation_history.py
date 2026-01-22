"""
Integration tests for conversation history in the agent pipeline.

Tests that conversation history flows correctly through the agent using LangGraph checkpointer.

Run with: pytest tests/backend/agent/test_conversation_history.py -v
"""

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.state import AgentState, format_conversation_for_prompt


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
        messages = [
            HumanMessage(content="Tell me about Acme"),
            AIMessage(content="Acme is..."),
        ]
        state: AgentState = {
            "question": "What about their contacts?",
            "messages": messages,
        }
        assert len(state["messages"]) == 2


class TestFormatConversationForPrompt:
    """Tests for format_conversation_for_prompt function."""

    def test_format_empty(self):
        """Test formatting empty history."""
        result = format_conversation_for_prompt([])
        assert result == ""

    def test_format_single_turn(self):
        """Test formatting a single turn."""
        messages = [HumanMessage(content="What is Acme's status?")]
        result = format_conversation_for_prompt(messages)
        assert "User: What is Acme's status?" in result

    def test_format_multi_turn(self):
        """Test formatting multiple turns."""
        messages = [
            HumanMessage(content="Tell me about Acme"),
            AIMessage(content="Acme Manufacturing is a mid-market account"),
            HumanMessage(content="What about their contacts?"),
        ]
        result = format_conversation_for_prompt(messages)

        assert "User: Tell me about Acme" in result
        assert "Assistant: Acme Manufacturing is a mid-market account" in result
        assert "User: What about their contacts?" in result

    def test_format_preserves_full_content(self):
        """Test that full content is preserved without truncation."""
        long_content = "A" * 500
        messages = [AIMessage(content=long_content)]
        result = format_conversation_for_prompt(messages)

        # Should preserve full content
        assert long_content in result

    def test_format_includes_all_messages(self):
        """Test that all messages are included."""
        messages = [HumanMessage(content=f"Question {i}") for i in range(10)]
        result = format_conversation_for_prompt(messages)

        # Should include all 10 messages
        for i in range(10):
            assert f"Question {i}" in result
