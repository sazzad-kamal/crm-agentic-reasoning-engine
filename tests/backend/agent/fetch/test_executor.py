"""
Tests for SQL query executor.

Covers execute_sql_plan, validate_sql, resolve_placeholders.
"""

import pytest
import duckdb

from backend.agent.fetch.executor import (
    execute_sql_plan,
    validate_sql,
    resolve_placeholders,
    SQLExecutionError,
    SQLExecutionStats,
    SQLValidationError,
)
from backend.agent.route.sql_planner import SQLPlan


# =============================================================================
# SQL Validation Tests
# =============================================================================


class TestValidateSql:
    """Tests for validate_sql function."""

    def test_validate_sql_allows_select(self):
        """SELECT queries are allowed."""
        validate_sql("SELECT * FROM companies")

    def test_validate_sql_allows_select_with_join(self):
        """JOIN queries are allowed."""
        validate_sql(
            "SELECT c.name, co.first_name FROM companies c "
            "JOIN contacts co ON c.company_id = co.company_id"
        )

    def test_validate_sql_allows_aggregate(self):
        """Aggregate queries are allowed."""
        validate_sql("SELECT COUNT(*), SUM(value) FROM opportunities GROUP BY company_id")

    def test_validate_sql_blocks_drop(self):
        """DROP is forbidden."""
        with pytest.raises(SQLValidationError, match="DROP"):
            validate_sql("DROP TABLE companies")

    def test_validate_sql_blocks_delete(self):
        """DELETE is forbidden."""
        with pytest.raises(SQLValidationError, match="DELETE"):
            validate_sql("DELETE FROM companies WHERE company_id = 'test'")

    def test_validate_sql_blocks_update(self):
        """UPDATE is forbidden."""
        with pytest.raises(SQLValidationError, match="UPDATE"):
            validate_sql("UPDATE companies SET name = 'hacked'")

    def test_validate_sql_blocks_insert(self):
        """INSERT is forbidden."""
        with pytest.raises(SQLValidationError, match="INSERT"):
            validate_sql("INSERT INTO companies VALUES ('test')")

    def test_validate_sql_blocks_alter(self):
        """ALTER is forbidden."""
        with pytest.raises(SQLValidationError, match="ALTER"):
            validate_sql("ALTER TABLE companies ADD COLUMN hacked TEXT")

    def test_validate_sql_blocks_truncate(self):
        """TRUNCATE is forbidden."""
        with pytest.raises(SQLValidationError, match="TRUNCATE"):
            validate_sql("TRUNCATE TABLE companies")

    def test_validate_sql_blocks_create(self):
        """CREATE is forbidden."""
        with pytest.raises(SQLValidationError, match="CREATE"):
            validate_sql("CREATE TABLE malicious (id INT)")

    def test_validate_sql_case_insensitive(self):
        """Keywords are blocked regardless of case."""
        with pytest.raises(SQLValidationError, match="DROP"):
            validate_sql("drop table companies")


# =============================================================================
# Placeholder Resolution Tests
# =============================================================================


class TestResolvePlaceholders:
    """Tests for resolve_placeholders function."""

    def test_resolve_company_id(self):
        """Resolves $company_id placeholder."""
        sql = "SELECT * FROM contacts WHERE company_id = $company_id"
        resolved = {"$company_id": "ACME-MFG"}
        result = resolve_placeholders(sql, resolved)
        assert result == "SELECT * FROM contacts WHERE company_id = 'ACME-MFG'"

    def test_resolve_multiple_placeholders(self):
        """Resolves multiple placeholders."""
        sql = "SELECT * FROM activities WHERE company_id = $company_id AND contact_id = $contact_id"
        resolved = {"$company_id": "ACME-MFG", "$contact_id": "C-ACME-ANNA"}
        result = resolve_placeholders(sql, resolved)
        assert "'ACME-MFG'" in result
        assert "'C-ACME-ANNA'" in result

    def test_resolve_escapes_quotes(self):
        """SQL injection via quotes is prevented."""
        sql = "SELECT * FROM companies WHERE name = $name"
        resolved = {"$name": "O'Reilly Media"}
        result = resolve_placeholders(sql, resolved)
        assert result == "SELECT * FROM companies WHERE name = 'O''Reilly Media'"

    def test_resolve_ignores_none_values(self):
        """None values are not replaced."""
        sql = "SELECT * FROM contacts WHERE company_id = $company_id"
        resolved = {"$company_id": None}
        result = resolve_placeholders(sql, resolved)
        assert result == sql  # Unchanged

    def test_resolve_preserves_unmatched_placeholders(self):
        """Unmatched placeholders remain in SQL."""
        sql = "SELECT * FROM activities WHERE company_id = $company_id AND contact_id = $contact_id"
        resolved = {"$company_id": "ACME-MFG"}
        result = resolve_placeholders(sql, resolved)
        assert "'ACME-MFG'" in result
        assert "$contact_id" in result


# =============================================================================
# SQL Execution Stats Tests
# =============================================================================


class TestSQLExecutionStats:
    """Tests for SQLExecutionStats class."""

    def test_stats_initial_values(self):
        """Stats start at zero."""
        stats = SQLExecutionStats()
        assert stats.total == 0
        assert stats.success == 0
        assert stats.failed == 0

    def test_stats_failed_property(self):
        """Failed is calculated from total - success."""
        stats = SQLExecutionStats()
        stats.total = 5
        stats.success = 3
        assert stats.failed == 2


# =============================================================================
# Execute SQL Plan Tests - In-Memory DuckDB
# =============================================================================


@pytest.fixture
def duckdb_connection():
    """Create in-memory DuckDB connection with test data."""
    conn = duckdb.connect(":memory:")

    # Create companies table
    conn.execute("""
        CREATE TABLE companies (
            company_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            status VARCHAR,
            plan VARCHAR,
            account_owner VARCHAR,
            health_flags VARCHAR,
            renewal_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO companies VALUES
        ('ACME-MFG', 'Acme Manufacturing', 'Active', 'Pro', 'jsmith', NULL, '2026-03-31'),
        ('BETA-TECH', 'Beta Tech Solutions', 'Active', 'Standard', 'amartin', NULL, '2026-02-15'),
        ('DELTA-HEALTH', 'Delta Health Clinics', 'Active', 'Standard', 'amartin', NULL, '2026-02-01')
    """)

    # Create contacts table
    conn.execute("""
        CREATE TABLE contacts (
            contact_id VARCHAR PRIMARY KEY,
            company_id VARCHAR,
            first_name VARCHAR,
            last_name VARCHAR,
            email VARCHAR,
            phone VARCHAR,
            job_title VARCHAR,
            role VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO contacts VALUES
        ('C-ACME-ANNA', 'ACME-MFG', 'Anna', 'Lopez', 'anna.lopez@acmemfg.com', '555-1234', 'VP Operations', 'Decision Maker'),
        ('C-ACME-JOE', 'ACME-MFG', 'Joe', 'Smith', 'joe.smith@acmemfg.com', '555-1235', 'IT Manager', 'Technical Contact'),
        ('C-BETA-LISA', 'BETA-TECH', 'Lisa', 'Ng', 'lisa.ng@betatech.io', '555-2234', 'CEO', 'Champion'),
        ('C-DELTA-ERIN', 'DELTA-HEALTH', 'Erin', 'Cho', 'erin.cho@deltahealth.org', '555-3234', 'CFO', 'Decision Maker')
    """)

    # Create opportunities table
    conn.execute("""
        CREATE TABLE opportunities (
            opportunity_id VARCHAR PRIMARY KEY,
            company_id VARCHAR,
            name VARCHAR,
            stage VARCHAR,
            value DECIMAL,
            owner VARCHAR,
            expected_close_date DATE,
            type VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO opportunities VALUES
        ('OPP-ACME-UPGRADE', 'ACME-MFG', 'Pro to Enterprise Upgrade', 'Proposal', 15000, 'jsmith', '2026-01-25', 'Expansion'),
        ('OPP-ACME-RENEWAL', 'ACME-MFG', 'Annual Renewal', 'Qualified', 24000, 'jsmith', '2026-04-15', 'Renewal'),
        ('OPP-BETA-NEW', 'BETA-TECH', 'New Rollout', 'Negotiation', 32000, 'amartin', '2026-02-15', 'New')
    """)

    # Create activities table
    conn.execute("""
        CREATE TABLE activities (
            activity_id VARCHAR PRIMARY KEY,
            company_id VARCHAR,
            contact_id VARCHAR,
            opportunity_id VARCHAR,
            type VARCHAR,
            subject VARCHAR,
            due_datetime TIMESTAMP
        )
    """)
    conn.execute("""
        INSERT INTO activities VALUES
        ('ACT-ACME-1', 'ACME-MFG', 'C-ACME-ANNA', 'OPP-ACME-UPGRADE', 'Call', 'Follow up on proposal', '2026-01-10 10:00:00'),
        ('ACT-BETA-1', 'BETA-TECH', 'C-BETA-LISA', 'OPP-BETA-NEW', 'Meeting', 'QBR', '2026-01-12 14:00:00')
    """)

    yield conn
    conn.close()


class TestExecuteSQLPlan:
    """Tests for execute_sql_plan function."""

    def test_execute_simple_select(self, duckdb_connection):
        """Executes simple SELECT query."""
        plan = SQLPlan(sql="SELECT * FROM companies", needs_rag=False)

        results, resolved, stats = execute_sql_plan(plan, duckdb_connection)

        assert "companies" in results
        assert len(results["companies"]) == 3
        assert stats.total == 1
        assert stats.success == 1

    def test_execute_filtered_query(self, duckdb_connection):
        """Executes filtered query."""
        plan = SQLPlan(
            sql="SELECT * FROM companies WHERE company_id = 'ACME-MFG'",
            needs_rag=True,
        )

        results, resolved, stats = execute_sql_plan(plan, duckdb_connection)

        assert "companies" in results
        assert len(results["companies"]) == 1
        assert results["companies"][0]["name"] == "Acme Manufacturing"

    def test_execute_with_order_by(self, duckdb_connection):
        """Executes query with ORDER BY."""
        plan = SQLPlan(
            sql="SELECT * FROM opportunities ORDER BY value DESC",
            needs_rag=False,
        )

        results, resolved, stats = execute_sql_plan(plan, duckdb_connection)

        assert len(results["opportunities"]) == 3
        # Highest value deal should be first (32000)
        assert results["opportunities"][0]["value"] == 32000

    def test_execute_with_join(self, duckdb_connection):
        """Executes query with JOIN."""
        plan = SQLPlan(
            sql="""
                SELECT c.*, co.name as company_name
                FROM contacts c
                JOIN companies co ON c.company_id = co.company_id
                WHERE co.name LIKE '%Acme%'
            """,
            needs_rag=False,
        )

        results, resolved, stats = execute_sql_plan(plan, duckdb_connection)

        assert stats.success == 1
        assert len(results["contacts"]) == 2  # Anna and Joe

    def test_execute_with_stage_filter(self, duckdb_connection):
        """Executes query with stage filter."""
        plan = SQLPlan(
            sql="SELECT * FROM opportunities WHERE stage = 'Negotiation'",
            needs_rag=False,
        )

        results, resolved, stats = execute_sql_plan(plan, duckdb_connection)

        assert len(results["opportunities"]) == 1
        assert results["opportunities"][0]["value"] == 32000

    def test_execute_resolves_company_id(self, duckdb_connection):
        """Query resolves $company_id from results."""
        plan = SQLPlan(
            sql="SELECT * FROM companies WHERE name LIKE '%Acme%'",
            needs_rag=True,
        )

        results, resolved, stats = execute_sql_plan(plan, duckdb_connection)

        assert resolved.get("$company_id") == "ACME-MFG"

    def test_execute_adds_limit_if_missing(self, duckdb_connection):
        """Adds LIMIT to queries without one."""
        plan = SQLPlan(sql="SELECT * FROM companies", needs_rag=False)

        # Execute with a small max_rows to verify LIMIT is added
        results, _, stats = execute_sql_plan(plan, duckdb_connection, max_rows=2)

        # Should return at most 2 rows due to added LIMIT
        assert len(results["companies"]) <= 2

    def test_execute_empty_plan(self, duckdb_connection):
        """Handles empty SQL plan."""
        plan = SQLPlan(sql="", needs_rag=False)

        results, resolved, stats = execute_sql_plan(plan, duckdb_connection)

        assert results == {}
        assert stats.total == 0


# =============================================================================
# Integration Tests - Real Data
# =============================================================================


class TestExecuteSQLPlanWithRealData:
    """Integration tests using real CRM data."""

    @pytest.fixture
    def real_connection(self):
        """Get connection to real CRM data."""
        from backend.agent.datastore.connection import get_connection

        return get_connection()

    def test_filter_contacts_by_role(self, real_connection):
        """Find contacts by role."""
        plan = SQLPlan(
            sql="SELECT * FROM contacts WHERE role = 'Decision Maker'",
            needs_rag=False,
        )

        results, _, stats = execute_sql_plan(plan, real_connection)

        assert stats.success == 1
        assert len(results["contacts"]) > 0
        for row in results["contacts"]:
            assert row["role"] == "Decision Maker"

    def test_get_activities(self, real_connection):
        """Get activities from real data."""
        plan = SQLPlan(sql="SELECT * FROM activities", needs_rag=False)

        results, _, stats = execute_sql_plan(plan, real_connection)

        assert stats.success == 1
        assert len(results["activities"]) > 0

    def test_filter_opportunities_by_stage(self, real_connection):
        """Filter opportunities by stage."""
        plan = SQLPlan(
            sql="SELECT * FROM opportunities WHERE stage NOT IN ('Closed Won', 'Closed Lost') ORDER BY value DESC",
            needs_rag=False,
        )

        results, _, stats = execute_sql_plan(plan, real_connection)

        assert stats.success == 1
        # All results should be open deals
        for row in results["opportunities"]:
            assert row["stage"] not in ["Closed Won", "Closed Lost"]
