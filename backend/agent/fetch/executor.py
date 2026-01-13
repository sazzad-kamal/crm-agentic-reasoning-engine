"""
SQL query executor.

Executes SQL queries against DuckDB.
"""

import logging
import re
from typing import Any

import duckdb

from backend.agent.route.sql_planner import SQLPlan

logger = logging.getLogger(__name__)


# Dangerous SQL keywords that should never appear in generated queries
FORBIDDEN_KEYWORDS = frozenset(
    {
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
        "EXEC",
        "EXECUTE",
    }
)


class SQLExecutionError(Exception):
    """Raised when SQL execution fails."""

    pass


class SQLValidationError(Exception):
    """Raised when SQL validation fails (e.g., dangerous keywords)."""

    pass


def validate_sql(sql: str) -> None:
    """
    Validate SQL query for safety.

    Raises:
        SQLValidationError: If SQL contains dangerous keywords
    """
    # Tokenize and check for forbidden keywords
    tokens = set(re.findall(r"\b[A-Z]+\b", sql.upper()))
    forbidden_found = tokens & FORBIDDEN_KEYWORDS

    if forbidden_found:
        raise SQLValidationError(f"SQL contains forbidden keywords: {forbidden_found}")


def resolve_placeholders(sql: str, resolved: dict[str, str]) -> str:
    """
    Resolve placeholders like $company_id and $contact_id in SQL.

    Args:
        sql: SQL query with placeholders
        resolved: Dict mapping placeholder names to resolved values

    Returns:
        SQL with placeholders replaced by actual values
    """
    result = sql
    for placeholder, value in resolved.items():
        if value is not None:
            # Escape single quotes in values for SQL safety
            safe_value = str(value).replace("'", "''")
            result = result.replace(placeholder, f"'{safe_value}'")
    return result


class SQLExecutionStats:
    """Statistics from SQL query execution."""

    def __init__(self) -> None:
        self.total: int = 0
        self.success: int = 0
        self.errors: dict[str, str] = {}  # purpose -> error message

    @property
    def failed(self) -> int:
        return self.total - self.success

    def get_error_summary(self) -> str | None:
        """Get combined error summary for retry feedback."""
        if not self.errors:
            return None
        return "; ".join(f"{purpose}: {error}" for purpose, error in self.errors.items())


def execute_sql_plan(
    plan: SQLPlan,
    conn: duckdb.DuckDBPyConnection,
    max_rows: int = 100,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], SQLExecutionStats]:
    """
    Execute SQLPlan query against DuckDB.

    Args:
        plan: SQLPlan containing raw SQL query
        conn: DuckDB connection
        max_rows: Maximum rows to return (safety limit)

    Returns:
        Tuple of (results dict, resolved placeholders dict, execution stats)

    Raises:
        SQLValidationError: If SQL contains dangerous keywords
    """
    results: dict[str, list[dict[str, Any]]] = {}
    resolved: dict[str, str] = {}
    stats = SQLExecutionStats()

    if not plan.sql or not plan.sql.strip():
        return results, resolved, stats

    stats.total = 1
    sql = plan.sql.strip()

    try:
        # Validate SQL for safety
        validate_sql(sql)

        # Add LIMIT if not present (safety)
        if "LIMIT" not in sql.upper():
            sql = f"{sql} LIMIT {max_rows}"

        logger.debug(f"Executing SQL: {sql[:100]}...")

        # Execute query
        result = conn.execute(sql)
        rows = result.fetchall()
        columns = [desc[0] for desc in result.description]

        # Convert to list of dicts
        data = [dict(zip(columns, row, strict=True)) for row in rows]

        # Enforce max_rows limit
        if len(data) > max_rows:
            logger.warning(f"Query returned {len(data)} rows, truncating to {max_rows}")
            data = data[:max_rows]

        # Determine result key from SQL (first table mentioned)
        result_key = _extract_table_from_sql(sql)
        results[result_key] = data
        stats.success = 1

        # Cache IDs for RAG filtering
        if data:
            first_row = data[0]
            for key in ["company_id", "contact_id", "opportunity_id"]:
                if key in first_row and first_row[key]:
                    resolved[f"${key}"] = str(first_row[key])

        logger.debug(f"Query returned {len(data)} rows")

    except SQLValidationError:
        raise

    except Exception as e:
        error_msg = str(e)
        logger.error(f"SQL execution failed: {error_msg}")
        stats.errors["query"] = error_msg
        results["query"] = []

    return results, resolved, stats


def _extract_table_from_sql(sql: str) -> str:
    """Extract main table name from SQL for result key."""
    sql_upper = sql.upper()

    # Try FROM clause
    match = re.search(r"\bFROM\s+(\w+)", sql_upper)
    if match:
        # Get actual case from original SQL
        start = match.start(1)
        end = match.end(1)
        return sql[start:end].lower()

    return "query"


__all__ = [
    "execute_sql_plan",
    "resolve_placeholders",
    "validate_sql",
    "SQLExecutionError",
    "SQLExecutionStats",
    "SQLValidationError",
]
