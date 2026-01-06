"""
Tests for conversation module.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestConversationCheckpointer:
    """Tests for conversation checkpointer."""

    def test_get_checkpointer_returns_saver(self):
        """Test get_checkpointer returns MemorySaver."""
        from backend.agent.nodes.support.session import get_checkpointer

        checkpointer = get_checkpointer()
        assert checkpointer is not None

    def test_get_session_state_not_found(self):
        """Test get_session_state returns None when not found."""
        from backend.agent.nodes.support.session import get_session_state

        result = get_session_state("nonexistent-session-id")
        assert result is None

    def test_get_session_state_exception(self):
        """Test get_session_state handles exceptions."""
        from backend.agent.nodes.support import session

        original_checkpointer = session._checkpointer

        with patch.object(session, "_checkpointer") as mock_cp:
            mock_cp.get.side_effect = Exception("Checkpointer error")

            result = session.get_session_state("test-session")
            assert result is None

        session._checkpointer = original_checkpointer

    def test_get_session_messages_not_found(self):
        """Test get_session_messages returns empty list when not found."""
        from backend.agent.nodes.support.session import get_session_messages

        result = get_session_messages("nonexistent-session-id")
        assert result == []

    def test_get_session_messages_exception(self):
        """Test get_session_messages handles exceptions."""
        from backend.agent.nodes.support import session

        original_checkpointer = session._checkpointer

        with patch.object(session, "_checkpointer") as mock_cp:
            mock_cp.get.side_effect = Exception("Checkpointer error")

            result = session.get_session_messages("test-session")
            assert result == []

        session._checkpointer = original_checkpointer

    def test_build_thread_config_with_session(self):
        """Test build_thread_config with session ID."""
        from backend.agent.nodes.support.session import build_thread_config

        config = build_thread_config("my-session")

        assert "configurable" in config
        assert config["configurable"]["thread_id"] == "my-session"

    def test_build_thread_config_without_session(self):
        """Test build_thread_config without session ID generates UUID."""
        from backend.agent.nodes.support.session import build_thread_config

        config = build_thread_config(None)

        assert "configurable" in config
        assert "thread_id" in config["configurable"]
        # UUID should be non-empty
        assert len(config["configurable"]["thread_id"]) > 0


class TestCompanyTools:
    """Tests for company tools module edge cases."""

    def test_tool_company_lookup_not_found(self):
        """Test tool_company_lookup when company not found."""
        from backend.agent.handlers import tool_company_lookup

        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = None
        mock_ds.get_company_name_matches.return_value = ["Similar Co"]

        result = tool_company_lookup("Unknown Company", datastore=mock_ds)

        assert result.data["found"] is False
        assert "close_matches" in result.data
        assert result.error is not None

    def test_tool_company_lookup_found_no_data(self):
        """Test tool_company_lookup when ID resolved but no data."""
        from backend.agent.handlers import tool_company_lookup

        mock_ds = MagicMock()
        mock_ds.resolve_company_id.return_value = "COMP001"
        mock_ds.get_company.return_value = None

        result = tool_company_lookup("Test Company", datastore=mock_ds)

        assert result.data["found"] is False
        assert result.error is not None

    def test_tool_search_contacts_empty_results(self):
        """Test tool_search_contacts with no results."""
        from backend.agent.handlers import tool_search_contacts

        mock_ds = MagicMock()
        mock_ds.search_contacts.return_value = []

        result = tool_search_contacts(query="nonexistent", datastore=mock_ds)

        assert result.data["count"] == 0
        assert result.data["contacts"] == []


class TestConfigEdgeCases:
    """Tests for config module edge cases."""

    def test_get_config_returns_config(self):
        """Test get_config returns a Config object."""
        from backend.agent.core.config import get_config

        config = get_config()
        assert config is not None
        assert hasattr(config, "llm_model")

    def test_config_has_expected_attributes(self):
        """Test config has expected attributes."""
        from backend.agent.core.config import get_config

        config = get_config()
        assert hasattr(config, "llm_temperature")
        assert hasattr(config, "llm_max_tokens")
        assert hasattr(config, "router_model")


class TestGraphModule:
    """Tests for graph module edge cases."""

    def test_build_agent_graph_returns_graph(self):
        """Test build_agent_graph returns a compiled graph."""
        from backend.agent.nodes.graph import build_agent_graph

        graph = build_agent_graph()
        assert graph is not None

    def test_run_agent_mock_mode(self):
        """Test run_agent in mock mode."""
        from backend.agent.nodes.graph import run_agent

        with patch("backend.agent.nodes.graph.agent_graph") as mock_graph:
            mock_graph.invoke.return_value = {
                "answer": "Test answer",
                "sources": [],
                "steps": [],
                "raw_data": {},
                "follow_up_suggestions": [],
                "mode_used": "data",
            }

            result = run_agent(question="What is the pipeline?", use_cache=False)

            assert "answer" in result


class TestStreamingModule:
    """Tests for streaming module edge cases."""

    def test_format_sse(self):
        """Test format_sse creates proper SSE format."""
        from backend.agent.nodes.support.streaming import _format_sse as format_sse

        result = format_sse("progress", {"step": "fetching"})

        assert "event: progress" in result
        assert "data:" in result
