"""
Tests for SQL query executor.

Covers execute_query_plan, validate_sql, resolve_placeholders,
and JOIN support for multi-table queries.
"""

import pytest
import duckdb

from backend.agent.fetch.executor import (
    execute_query_plan,
    validate_sql,
    resolve_placeholders,
    SQLExecutionError,
    SQLExecutionStats,
    SQLValidationError,
)
from backend.agent.route.query_planner import QueryPlan, SQLQuery


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
# Execute Query Plan Tests - In-Memory DuckDB
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
            industry VARCHAR,
            renewal_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO companies VALUES
        ('ACME-MFG', 'Acme Manufacturing', 'Active', 'Pro', 'Manufacturing', '2026-03-31'),
        ('BETA-TECH', 'Beta Tech Solutions', 'Active', 'Standard', 'Technology', '2026-02-15'),
        ('DELTA-HEALTH', 'Delta Health Clinics', 'Active', 'Standard', 'Healthcare', '2026-02-01')
    """)

    # Create contacts table
    conn.execute("""
        CREATE TABLE contacts (
            contact_id VARCHAR PRIMARY KEY,
            company_id VARCHAR,
            first_name VARCHAR,
            last_name VARCHAR,
            email VARCHAR,
            role VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO contacts VALUES
        ('C-ACME-ANNA', 'ACME-MFG', 'Anna', 'Lopez', 'anna.lopez@acmemfg.com', 'Decision Maker'),
        ('C-ACME-JOE', 'ACME-MFG', 'Joe', 'Smith', 'joe.smith@acmemfg.com', 'Technical Contact'),
        ('C-BETA-LISA', 'BETA-TECH', 'Lisa', 'Ng', 'lisa.ng@betatech.io', 'Champion'),
        ('C-DELTA-ERIN', 'DELTA-HEALTH', 'Erin', 'Cho', 'erin.cho@deltahealth.org', 'Decision Maker')
    """)

    # Create opportunities table
    conn.execute("""
        CREATE TABLE opportunities (
            opportunity_id VARCHAR PRIMARY KEY,
            company_id VARCHAR,
            primary_contact_id VARCHAR,
            name VARCHAR,
            stage VARCHAR,
            value DECIMAL,
            expected_close_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO opportunities VALUES
        ('OPP-ACME-UPGRADE', 'ACME-MFG', 'C-ACME-ANNA', 'Pro to Enterprise Upgrade', 'Proposal', 15000, '2026-01-25'),
        ('OPP-ACME-RENEWAL', 'ACME-MFG', 'C-ACME-JOE', 'Annual Renewal', 'Qualified', 24000, '2026-04-15'),
        ('OPP-BETA-NEW', 'BETA-TECH', 'C-BETA-LISA', 'New Rollout', 'Negotiation', 32000, '2026-02-15')
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
            status VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO activities VALUES
        ('ACT-ACME-1', 'ACME-MFG', 'C-ACME-ANNA', 'OPP-ACME-UPGRADE', 'Call', 'Follow up on proposal', 'Scheduled'),
        ('ACT-BETA-1', 'BETA-TECH', 'C-BETA-LISA', 'OPP-BETA-NEW', 'Meeting', 'QBR', 'Scheduled')
    """)

    yield conn
    conn.close()


class TestExecuteQueryPlan:
    """Tests for execute_query_plan function."""

    def test_execute_simple_select(self, duckdb_connection):
        """Executes simple SELECT query."""
        plan = QueryPlan(
            queries=[SQLQuery(sql="SELECT * FROM companies", purpose="all_companies")],
            needs_account_rag=False,
        )

        results, resolved, stats = execute_query_plan(plan, duckdb_connection)

        assert "all_companies" in results
        assert len(results["all_companies"]) == 3
        assert stats.total == 1
        assert stats.success == 1

    def test_execute_join_companies_contacts(self, duckdb_connection):
        """Executes JOIN between companies and contacts."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="""
                    SELECT c.name, co.first_name, co.last_name, co.role
                    FROM companies c
                    JOIN contacts co ON c.company_id = co.company_id
                    WHERE c.company_id = 'ACME-MFG'
                    """,
                    purpose="company_contacts",
                )
            ],
            needs_account_rag=True,
        )

        results, resolved, stats = execute_query_plan(plan, duckdb_connection)

        assert "company_contacts" in results
        assert len(results["company_contacts"]) == 2  # Anna and Joe
        assert any(r["first_name"] == "Anna" for r in results["company_contacts"])
        assert any(r["first_name"] == "Joe" for r in results["company_contacts"])

    def test_execute_join_contacts_opportunities(self, duckdb_connection):
        """Executes JOIN between contacts and opportunities."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="""
                    SELECT o.name as opp_name, o.value, c.first_name, c.last_name
                    FROM opportunities o
                    JOIN contacts c ON o.primary_contact_id = c.contact_id
                    WHERE o.stage = 'Negotiation'
                    """,
                    purpose="deals_with_contacts",
                )
            ],
            needs_account_rag=False,
        )

        results, resolved, stats = execute_query_plan(plan, duckdb_connection)

        assert len(results["deals_with_contacts"]) == 1
        assert results["deals_with_contacts"][0]["first_name"] == "Lisa"
        assert results["deals_with_contacts"][0]["value"] == 32000

    def test_execute_multi_table_join(self, duckdb_connection):
        """Executes JOIN across companies, contacts, and opportunities."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="""
                    SELECT
                        comp.name as company,
                        cont.first_name || ' ' || cont.last_name as contact,
                        opp.name as opportunity,
                        opp.value
                    FROM companies comp
                    JOIN opportunities opp ON comp.company_id = opp.company_id
                    JOIN contacts cont ON opp.primary_contact_id = cont.contact_id
                    ORDER BY opp.value DESC
                    """,
                    purpose="pipeline_with_contacts",
                )
            ],
            needs_account_rag=False,
        )

        results, resolved, stats = execute_query_plan(plan, duckdb_connection)

        assert len(results["pipeline_with_contacts"]) == 3
        # Highest value deal should be first
        assert results["pipeline_with_contacts"][0]["value"] == 32000
        assert "Lisa" in results["pipeline_with_contacts"][0]["contact"]

    def test_execute_join_with_activities(self, duckdb_connection):
        """Executes JOIN including activities table."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="""
                    SELECT
                        co.first_name || ' ' || co.last_name as contact,
                        a.type,
                        a.subject,
                        o.name as opportunity
                    FROM activities a
                    JOIN contacts co ON a.contact_id = co.contact_id
                    JOIN opportunities o ON a.opportunity_id = o.opportunity_id
                    """,
                    purpose="activities_with_context",
                )
            ],
            needs_account_rag=False,
        )

        results, resolved, stats = execute_query_plan(plan, duckdb_connection)

        assert len(results["activities_with_context"]) == 2
        assert any("Anna Lopez" in r["contact"] for r in results["activities_with_context"])
        assert any("Lisa Ng" in r["contact"] for r in results["activities_with_context"])

    def test_execute_resolves_company_id_placeholder(self, duckdb_connection):
        """First query resolves $company_id for subsequent queries."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="SELECT company_id, name FROM companies WHERE name LIKE '%Acme%'",
                    purpose="find_company",
                ),
                SQLQuery(
                    sql="SELECT * FROM contacts WHERE company_id = $company_id",
                    purpose="company_contacts",
                ),
            ],
            needs_account_rag=True,
        )

        results, resolved, stats = execute_query_plan(plan, duckdb_connection)

        assert resolved.get("$company_id") == "ACME-MFG"
        assert len(results["company_contacts"]) == 2  # Anna and Joe

    def test_execute_resolves_contact_id_placeholder(self, duckdb_connection):
        """First query resolves $contact_id for subsequent queries."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="SELECT contact_id, first_name FROM contacts WHERE first_name = 'Anna'",
                    purpose="find_contact",
                ),
                SQLQuery(
                    sql="SELECT * FROM activities WHERE contact_id = $contact_id",
                    purpose="contact_activities",
                ),
            ],
            needs_account_rag=True,
        )

        results, resolved, stats = execute_query_plan(plan, duckdb_connection)

        assert resolved.get("$contact_id") == "C-ACME-ANNA"
        assert len(results["contact_activities"]) == 1

    def test_execute_resolves_opportunity_id_placeholder(self, duckdb_connection):
        """First query resolves $opportunity_id for subsequent queries."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="SELECT opportunity_id, name FROM opportunities WHERE stage = 'Negotiation'",
                    purpose="find_opportunity",
                ),
                SQLQuery(
                    sql="SELECT * FROM activities WHERE opportunity_id = $opportunity_id",
                    purpose="opp_activities",
                ),
            ],
            needs_account_rag=True,
        )

        results, resolved, stats = execute_query_plan(plan, duckdb_connection)

        assert resolved.get("$opportunity_id") == "OPP-BETA-NEW"
        assert len(results["opp_activities"]) == 1

    def test_execute_aggregate_query(self, duckdb_connection):
        """Executes aggregate query with GROUP BY."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="""
                    SELECT company_id, COUNT(*) as opp_count, SUM(value) as total_value
                    FROM opportunities
                    GROUP BY company_id
                    ORDER BY total_value DESC
                    """,
                    purpose="pipeline_by_company",
                )
            ],
            needs_account_rag=False,
        )

        results, resolved, stats = execute_query_plan(plan, duckdb_connection)

        assert len(results["pipeline_by_company"]) == 2
        # ACME has 2 opps totaling 39000
        acme = next(r for r in results["pipeline_by_company"] if r["company_id"] == "ACME-MFG")
        assert acme["opp_count"] == 2
        assert acme["total_value"] == 39000

    def test_execute_adds_limit_if_missing(self, duckdb_connection):
        """Adds LIMIT to queries without one."""
        plan = QueryPlan(
            queries=[SQLQuery(sql="SELECT * FROM companies", purpose="test")],
            needs_account_rag=False,
        )

        # Execute with a small max_rows to verify LIMIT is added
        results, _, stats = execute_query_plan(plan, duckdb_connection, max_rows=2)

        # Should return at most 2 rows due to added LIMIT
        assert len(results["test"]) <= 2

    def test_execute_respects_existing_limit(self, duckdb_connection):
        """Preserves existing LIMIT in query."""
        plan = QueryPlan(
            queries=[SQLQuery(sql="SELECT * FROM companies LIMIT 1", purpose="test")],
            needs_account_rag=False,
        )

        results, _, stats = execute_query_plan(plan, duckdb_connection, max_rows=100)

        # Original LIMIT 1 should be preserved
        assert len(results["test"]) == 1

    def test_execute_blocks_dangerous_sql(self, duckdb_connection):
        """Raises exception for dangerous SQL."""
        plan = QueryPlan(
            queries=[SQLQuery(sql="DROP TABLE companies", purpose="malicious")],
            needs_account_rag=False,
        )

        with pytest.raises(SQLValidationError, match="DROP"):
            execute_query_plan(plan, duckdb_connection)

    def test_execute_handles_failed_query_gracefully(self, duckdb_connection):
        """Failed query returns empty result, doesn't fail entire plan."""
        plan = QueryPlan(
            queries=[
                SQLQuery(sql="SELECT * FROM nonexistent_table", purpose="bad_query"),
                SQLQuery(sql="SELECT * FROM companies", purpose="good_query"),
            ],
            needs_account_rag=False,
        )

        results, _, stats = execute_query_plan(plan, duckdb_connection)

        assert results["bad_query"] == []  # Empty result for failed query
        assert len(results["good_query"]) == 3  # Good query still works
        assert stats.total == 2
        assert stats.success == 1
        assert stats.failed == 1


# =============================================================================
# Integration Tests - Real Data
# =============================================================================


class TestExecuteQueryPlanWithRealData:
    """Integration tests using real CRM data."""

    @pytest.fixture
    def real_connection(self):
        """Get connection to real CRM data."""
        from backend.agent.datastore.connection import get_connection

        return get_connection()

    def test_join_decision_makers_with_opportunities(self, real_connection):
        """Find decision makers and their open opportunities."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="""
                    SELECT
                        c.first_name || ' ' || c.last_name as contact,
                        c.role,
                        comp.name as company,
                        o.name as opportunity,
                        o.value,
                        o.stage
                    FROM contacts c
                    JOIN companies comp ON c.company_id = comp.company_id
                    JOIN opportunities o ON o.primary_contact_id = c.contact_id
                    WHERE c.role = 'Decision Maker'
                    ORDER BY o.value DESC
                    """,
                    purpose="decision_makers_pipeline",
                )
            ],
            needs_account_rag=False,
        )

        results, _, stats = execute_query_plan(plan, real_connection)

        assert stats.success == 1
        assert len(results["decision_makers_pipeline"]) > 0
        # All results should have Decision Maker role
        for row in results["decision_makers_pipeline"]:
            assert row["role"] == "Decision Maker"

    def test_join_activities_with_full_context(self, real_connection):
        """Get activities with company, contact, and opportunity context."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="""
                    SELECT
                        a.type,
                        a.subject,
                        a.status,
                        comp.name as company,
                        c.first_name || ' ' || c.last_name as contact,
                        o.name as opportunity
                    FROM activities a
                    LEFT JOIN companies comp ON a.company_id = comp.company_id
                    LEFT JOIN contacts c ON a.contact_id = c.contact_id
                    LEFT JOIN opportunities o ON a.opportunity_id = o.opportunity_id
                    LIMIT 10
                    """,
                    purpose="activities_full_context",
                )
            ],
            needs_account_rag=False,
        )

        results, _, stats = execute_query_plan(plan, real_connection)

        assert stats.success == 1
        assert len(results["activities_full_context"]) <= 10

    def test_pipeline_summary_with_contacts(self, real_connection):
        """Get pipeline summary with primary contact names."""
        plan = QueryPlan(
            queries=[
                SQLQuery(
                    sql="""
                    SELECT
                        comp.name as company,
                        o.name as deal,
                        o.stage,
                        o.value,
                        c.first_name || ' ' || c.last_name as primary_contact,
                        o.expected_close_date
                    FROM opportunities o
                    JOIN companies comp ON o.company_id = comp.company_id
                    LEFT JOIN contacts c ON o.primary_contact_id = c.contact_id
                    WHERE o.stage NOT IN ('Closed Won', 'Closed Lost')
                    ORDER BY o.expected_close_date
                    """,
                    purpose="pipeline_with_contacts",
                )
            ],
            needs_account_rag=False,
        )

        results, _, stats = execute_query_plan(plan, real_connection)

        assert stats.success == 1
        # Verify we get structured data with contacts
        for row in results["pipeline_with_contacts"]:
            assert "company" in row
            assert "deal" in row
            assert "primary_contact" in row
