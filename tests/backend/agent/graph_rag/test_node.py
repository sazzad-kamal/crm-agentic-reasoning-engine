"""Tests for Neo4j graph RAG node."""

from unittest.mock import MagicMock, patch

from backend.agent.graph_rag.guard import CypherGuardResult
from backend.agent.graph_rag.node import graph_node
from backend.agent.graph_rag.planner import CypherPlan


class TestGraphNode:
    """Tests for graph_node function."""

    def test_successful_graph_query(self):
        mock_plan = CypherPlan(cypher="MATCH (c:Company) RETURN c.name", explanation="test")
        mock_guard = CypherGuardResult(is_safe=True, cypher="MATCH (c:Company) RETURN c.name LIMIT 1000")
        mock_records = [{"name": "Acme Corp"}]

        with patch("backend.agent.graph_rag.node.get_cypher_plan", return_value=mock_plan), \
             patch("backend.agent.graph_rag.node.validate_cypher", return_value=mock_guard), \
             patch("backend.agent.graph_rag.node.get_driver") as mock_driver, \
             patch("backend.agent.graph_rag.node.execute_cypher", return_value=(mock_records, None)):

            state = {"question": "Show relationships for Acme", "messages": []}
            result = graph_node(state)

        assert result["sql_results"]["graph_data"] == [{"name": "Acme Corp"}]
        assert result["sql_results"]["_source"] == "neo4j"

    def test_unsafe_cypher_blocked(self):
        mock_plan = CypherPlan(cypher="DELETE (n)", explanation="test")
        mock_guard = CypherGuardResult(is_safe=False, cypher="DELETE (n)", reason="Forbidden: DELETE")

        with patch("backend.agent.graph_rag.node.get_cypher_plan", return_value=mock_plan), \
             patch("backend.agent.graph_rag.node.validate_cypher", return_value=mock_guard):

            state = {"question": "Delete everything", "messages": []}
            result = graph_node(state)

        assert result["sql_results"]["graph_data"] == []
        assert "blocked" in result["sql_results"]["error"].lower()

    def test_execution_error_handled(self):
        mock_plan = CypherPlan(cypher="MATCH (c) RETURN c", explanation="test")
        mock_guard = CypherGuardResult(is_safe=True, cypher="MATCH (c) RETURN c LIMIT 1000")

        with patch("backend.agent.graph_rag.node.get_cypher_plan", return_value=mock_plan), \
             patch("backend.agent.graph_rag.node.validate_cypher", return_value=mock_guard), \
             patch("backend.agent.graph_rag.node.get_driver"), \
             patch("backend.agent.graph_rag.node.execute_cypher", return_value=([], "Neo4j unreachable")):

            state = {"question": "Show graph data", "messages": []}
            result = graph_node(state)

        assert result["sql_results"]["error"] == "Neo4j unreachable"

    def test_exception_handled_gracefully(self):
        with patch("backend.agent.graph_rag.node.get_cypher_plan", side_effect=Exception("LLM down")):
            state = {"question": "Test", "messages": []}
            result = graph_node(state)

        assert result["sql_results"]["graph_data"] == []
        assert "LLM down" in result["sql_results"]["error"]
