"""Tests for SQL query executor."""

import pytest
import duckdb

from backend.agent.fetch.sql.executor import execute_sql


@pytest.fixture
def duckdb_connection():
    """Create in-memory DuckDB connection with test data."""
    conn = duckdb.connect(":memory:")

    conn.execute("""
        CREATE TABLE companies (
            company_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            status VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO companies VALUES
        ('ACME-MFG', 'Acme Manufacturing', 'Active'),
        ('BETA-TECH', 'Beta Tech Solutions', 'Active'),
        ('DELTA-HEALTH', 'Delta Health Clinics', 'Active')
    """)

    conn.execute("""
        CREATE TABLE contacts (
            contact_id VARCHAR PRIMARY KEY,
            company_id VARCHAR,
            first_name VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO contacts VALUES
        ('C-ACME-ANNA', 'ACME-MFG', 'Anna'),
        ('C-ACME-JOE', 'ACME-MFG', 'Joe')
    """)

    conn.execute("""
        CREATE TABLE opportunities (
            opportunity_id VARCHAR PRIMARY KEY,
            company_id VARCHAR,
            value DECIMAL
        )
    """)
    conn.execute("""
        INSERT INTO opportunities VALUES
        ('OPP-1', 'ACME-MFG', 15000),
        ('OPP-2', 'BETA-TECH', 32000)
    """)

    yield conn
    conn.close()


class TestExecuteSql:
    """Tests for execute_sql function."""

    def test_simple_select(self, duckdb_connection):
        """Executes simple SELECT query."""
        rows, error = execute_sql("SELECT * FROM companies", duckdb_connection)

        assert error is None
        assert len(rows) == 3
        assert rows[0]["company_id"] == "ACME-MFG"

    def test_filtered_query(self, duckdb_connection):
        """Executes filtered query."""
        rows, error = execute_sql(
            "SELECT * FROM companies WHERE company_id = 'ACME-MFG'",
            duckdb_connection,
        )

        assert error is None
        assert len(rows) == 1
        assert rows[0]["name"] == "Acme Manufacturing"

    def test_adds_limit(self, duckdb_connection):
        """Adds LIMIT to queries without one."""
        rows, error = execute_sql("SELECT * FROM companies", duckdb_connection, max_rows=2)

        assert error is None
        assert len(rows) <= 2

    def test_handles_error(self, duckdb_connection):
        """Returns error message on failure."""
        rows, error = execute_sql("SELECT * FROM nonexistent_table", duckdb_connection)

        assert rows == []
        assert error is not None
        assert "nonexistent_table" in error.lower() or "does not exist" in error.lower()

    def test_join_query(self, duckdb_connection):
        """Executes JOIN query."""
        sql = """
            SELECT c.*, co.name as company_name
            FROM contacts c
            JOIN companies co ON c.company_id = co.company_id
        """

        rows, error = execute_sql(sql, duckdb_connection)

        assert error is None
        assert len(rows) == 2


class TestExecuteSqlWithRealData:
    """Integration tests using real CRM data."""

    @pytest.fixture
    def real_connection(self):
        """Get connection to real CRM data."""
        from backend.agent.fetch.sql.connection import get_connection
        return get_connection()

    def test_filter_contacts_by_role(self, real_connection):
        """Find contacts by role."""
        rows, error = execute_sql(
            "SELECT * FROM contacts WHERE role = 'Decision Maker'",
            real_connection,
        )

        assert error is None
        assert len(rows) > 0
        for row in rows:
            assert row["role"] == "Decision Maker"

    def test_get_activities(self, real_connection):
        """Get activities from real data."""
        rows, error = execute_sql("SELECT * FROM activities", real_connection)

        assert error is None
        assert len(rows) > 0
