"""Tests for SQL Safety Guard."""

import pytest

from backend.agent.sql.guard import (
    MAX_ROWS,
    SQLGuardResult,
    validate_sql,
)


class TestValidateSql:
    """Tests for validate_sql function."""

    # =========================================================================
    # Valid SELECT queries
    # =========================================================================

    def test_simple_select_is_safe(self):
        result = validate_sql("SELECT * FROM contacts")
        assert result.is_safe is True
        assert result.error is None

    def test_select_with_where_is_safe(self):
        result = validate_sql("SELECT name FROM contacts WHERE id = 1")
        assert result.is_safe is True

    def test_select_with_join_is_safe(self):
        result = validate_sql(
            "SELECT c.name, o.value FROM contacts c JOIN opportunities o ON c.id = o.contact_id"
        )
        assert result.is_safe is True

    def test_select_with_aggregation_is_safe(self):
        result = validate_sql(
            "SELECT COUNT(*), SUM(value) FROM opportunities GROUP BY status"
        )
        assert result.is_safe is True

    def test_select_with_subquery_is_safe(self):
        result = validate_sql(
            "SELECT * FROM contacts WHERE id IN (SELECT contact_id FROM opportunities)"
        )
        assert result.is_safe is True

    # =========================================================================
    # Forbidden statement types
    # =========================================================================

    def test_insert_is_blocked(self):
        result = validate_sql("INSERT INTO contacts (name) VALUES ('test')")
        assert result.is_safe is False
        assert "Forbidden SQL operation" in result.error or "Only SELECT" in result.error

    def test_update_is_blocked(self):
        result = validate_sql("UPDATE contacts SET name = 'test' WHERE id = 1")
        assert result.is_safe is False

    def test_delete_is_blocked(self):
        result = validate_sql("DELETE FROM contacts WHERE id = 1")
        assert result.is_safe is False

    def test_drop_table_is_blocked(self):
        result = validate_sql("DROP TABLE contacts")
        assert result.is_safe is False

    def test_create_table_is_blocked(self):
        result = validate_sql("CREATE TABLE evil (id INT)")
        assert result.is_safe is False

    def test_alter_table_is_blocked(self):
        result = validate_sql("ALTER TABLE contacts ADD COLUMN evil TEXT")
        assert result.is_safe is False

    def test_truncate_is_blocked(self):
        # TRUNCATE is not a SELECT, so it gets blocked
        result = validate_sql("TRUNCATE TABLE contacts")
        assert result.is_safe is False

    # =========================================================================
    # Forbidden functions
    # =========================================================================

    def test_read_csv_function_is_blocked(self):
        # DuckDB's read_csv function could expose file system
        result = validate_sql("SELECT * FROM read_csv('/etc/passwd')")
        assert result.is_safe is False
        assert "Forbidden function" in result.error

    def test_attach_is_blocked(self):
        result = validate_sql("ATTACH DATABASE '/tmp/evil.db'")
        assert result.is_safe is False

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_empty_query_is_blocked(self):
        result = validate_sql("")
        assert result.is_safe is False
        assert "Empty" in result.error

    def test_whitespace_only_is_blocked(self):
        result = validate_sql("   ")
        assert result.is_safe is False

    def test_invalid_sql_is_blocked(self):
        result = validate_sql("NOT VALID SQL AT ALL")
        assert result.is_safe is False

    # =========================================================================
    # LIMIT enforcement
    # =========================================================================

    def test_adds_limit_when_missing(self):
        result = validate_sql("SELECT * FROM contacts")
        assert result.is_safe is True
        assert f"LIMIT {MAX_ROWS}" in result.sql.upper()

    def test_preserves_existing_limit(self):
        result = validate_sql("SELECT * FROM contacts LIMIT 10")
        assert result.is_safe is True
        # Should not add another LIMIT
        assert result.sql.upper().count("LIMIT") == 1

    def test_respects_smaller_limit(self):
        result = validate_sql("SELECT * FROM contacts LIMIT 5")
        assert result.is_safe is True
        assert "LIMIT 5" in result.sql.upper() or "LIMIT 5" in result.sql

    # =========================================================================
    # SQL injection attempts
    # =========================================================================

    def test_semicolon_injection_blocked(self):
        result = validate_sql("SELECT * FROM contacts; DROP TABLE contacts;")
        assert result.is_safe is False

    def test_union_select_safe(self):
        # UNION SELECT is actually valid SQL and safe
        result = validate_sql(
            "SELECT name FROM contacts UNION SELECT name FROM companies"
        )
        assert result.is_safe is True

    def test_comment_injection(self):
        result = validate_sql("SELECT * FROM contacts -- WHERE id = 1")
        assert result.is_safe is True  # Comments are fine in SELECT


class TestSQLGuardResult:
    """Tests for SQLGuardResult dataclass."""

    def test_safe_result(self):
        result = SQLGuardResult(is_safe=True, sql="SELECT 1")
        assert result.is_safe is True
        assert result.error is None

    def test_unsafe_result(self):
        result = SQLGuardResult(is_safe=False, sql="DROP TABLE x", error="Forbidden")
        assert result.is_safe is False
        assert result.error == "Forbidden"
