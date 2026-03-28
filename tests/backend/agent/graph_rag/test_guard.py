"""Tests for Neo4j Cypher safety guard."""

import pytest

from backend.agent.graph_rag.guard import MAX_RESULTS, validate_cypher


class TestValidateCypher:
    """Tests for validate_cypher function."""

    # --- Valid read queries ---

    def test_simple_match_passes(self):
        result = validate_cypher("MATCH (c:Company) RETURN c.name")
        assert result.is_safe is True

    def test_match_with_where_passes(self):
        result = validate_cypher(
            "MATCH (c:Company)-[:HAS_CONTACT]->(ct:Contact) "
            "WHERE c.status = 'Active' RETURN ct.first_name"
        )
        assert result.is_safe is True

    def test_optional_match_passes(self):
        result = validate_cypher(
            "MATCH (c:Company) OPTIONAL MATCH (c)-[:HAS_OPPORTUNITY]->(o) RETURN c, o"
        )
        assert result.is_safe is True

    def test_with_clause_passes(self):
        result = validate_cypher(
            "MATCH (c:Company) WITH c, count(*) AS cnt RETURN c.name, cnt"
        )
        assert result.is_safe is True

    # --- Forbidden write operations ---

    def test_create_blocked(self):
        result = validate_cypher("CREATE (n:Company {name: 'Evil Corp'})")
        assert result.is_safe is False
        assert "CREATE" in result.reason

    def test_delete_blocked(self):
        result = validate_cypher("MATCH (n) DELETE n")
        assert result.is_safe is False
        assert "DELETE" in result.reason

    def test_detach_delete_blocked(self):
        result = validate_cypher("MATCH (n) DETACH DELETE n")
        assert result.is_safe is False

    def test_set_blocked(self):
        result = validate_cypher("MATCH (c:Company) SET c.status = 'Inactive'")
        assert result.is_safe is False
        assert "SET" in result.reason

    def test_merge_blocked(self):
        result = validate_cypher("MERGE (c:Company {name: 'Test'})")
        assert result.is_safe is False

    def test_remove_blocked(self):
        result = validate_cypher("MATCH (c:Company) REMOVE c.notes")
        assert result.is_safe is False

    def test_drop_blocked(self):
        result = validate_cypher("DROP INDEX ON :Company(name)")
        assert result.is_safe is False

    # --- Edge cases ---

    def test_empty_query_rejected(self):
        result = validate_cypher("")
        assert result.is_safe is False

    def test_whitespace_only_rejected(self):
        result = validate_cypher("   ")
        assert result.is_safe is False

    # --- LIMIT auto-injection ---

    def test_auto_adds_limit(self):
        result = validate_cypher("MATCH (c:Company) RETURN c.name")
        assert result.is_safe is True
        assert f"LIMIT {MAX_RESULTS}" in result.cypher

    def test_preserves_existing_limit(self):
        result = validate_cypher("MATCH (c:Company) RETURN c.name LIMIT 10")
        assert result.is_safe is True
        assert f"LIMIT {MAX_RESULTS}" not in result.cypher
        assert "LIMIT 10" in result.cypher
