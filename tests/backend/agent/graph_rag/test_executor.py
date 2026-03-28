"""Tests for Neo4j Cypher executor."""

from unittest.mock import MagicMock

from backend.agent.graph_rag.executor import execute_cypher


class TestExecuteCypher:
    """Tests for execute_cypher function."""

    def test_successful_execution(self):
        mock_record1 = MagicMock()
        mock_record1.__iter__ = lambda self: iter([("name", "Acme Corp")])
        mock_record1.keys.return_value = ["name"]
        mock_record1.__getitem__ = lambda self, key: "Acme Corp"

        # Simpler approach: make records return dicts directly
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([{"name": "Acme Corp"}, {"name": "Globex"}])

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = lambda self: self
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        records, error = execute_cypher("MATCH (c:Company) RETURN c.name", mock_driver)

        assert error is None
        assert len(records) == 2

    def test_execution_error(self):
        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("Connection lost")
        mock_session.__enter__ = lambda self: self
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        records, error = execute_cypher("MATCH (c:Company) RETURN c", mock_driver)

        assert records == []
        assert "Connection lost" in error

    def test_empty_results(self):
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = lambda self: self
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        records, error = execute_cypher("MATCH (c:Company) RETURN c", mock_driver)

        assert error is None
        assert records == []
