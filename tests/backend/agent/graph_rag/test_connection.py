"""Tests for Neo4j connection manager."""

from unittest.mock import MagicMock, patch

from backend.agent.graph_rag.connection import close_driver, get_driver


class TestGetDriver:
    """Tests for get_driver function."""

    @patch("backend.agent.graph_rag.connection.GraphDatabase")
    @patch("backend.agent.graph_rag.connection._load_csv_data")
    def test_creates_driver(self, mock_load, mock_gdb):
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        # Reset thread-local state
        import backend.agent.graph_rag.connection as conn_module
        if hasattr(conn_module._thread_local, "driver"):
            conn_module._thread_local.driver = None

        driver = get_driver()

        mock_gdb.driver.assert_called_once()
        mock_load.assert_called_once_with(mock_driver)
        assert driver == mock_driver

        # Cleanup
        conn_module._thread_local.driver = None

    @patch("backend.agent.graph_rag.connection.GraphDatabase")
    @patch("backend.agent.graph_rag.connection._load_csv_data")
    def test_returns_cached_driver(self, mock_load, mock_gdb):
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver

        import backend.agent.graph_rag.connection as conn_module
        if hasattr(conn_module._thread_local, "driver"):
            conn_module._thread_local.driver = None

        driver1 = get_driver()
        driver2 = get_driver()

        # Should only create once
        assert mock_gdb.driver.call_count == 1
        assert driver1 == driver2

        conn_module._thread_local.driver = None


class TestCloseDriver:
    """Tests for close_driver function."""

    def test_close_when_driver_exists(self):
        import backend.agent.graph_rag.connection as conn_module
        mock_driver = MagicMock()
        conn_module._thread_local.driver = mock_driver

        close_driver()

        mock_driver.close.assert_called_once()
        assert conn_module._thread_local.driver is None

    def test_close_when_no_driver(self):
        import backend.agent.graph_rag.connection as conn_module
        conn_module._thread_local.driver = None
        close_driver()  # Should not raise
