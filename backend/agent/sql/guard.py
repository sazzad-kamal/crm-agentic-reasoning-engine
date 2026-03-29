"""SQL Safety Guard - validates SQL before execution.

Prevents dangerous operations like INSERT, UPDATE, DELETE, DROP, etc.
Ensures only SELECT queries are executed.
"""

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

# Forbidden SQL statement types
FORBIDDEN_STATEMENTS = frozenset({
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Grant,
})

# Forbidden function names (DuckDB-specific dangerous functions)
FORBIDDEN_FUNCTIONS = frozenset({
    "copy",
    "export",
    "attach",
    "detach",
    "load",
    "install",
    "write_csv",
    "write_parquet",
})

# Forbidden table-valued functions (parsed as special expression types)
# These allow reading arbitrary files from the filesystem
FORBIDDEN_TABLE_FUNCTIONS = frozenset({
    exp.ReadCSV,
})

# Maximum allowed result rows
MAX_ROWS = 1000


@dataclass
class SQLGuardResult:
    """Result of SQL validation."""

    is_safe: bool
    sql: str
    error: str | None = None


def validate_sql(sql: str) -> SQLGuardResult:
    """Validate SQL for safety before execution.

    Args:
        sql: The SQL query to validate

    Returns:
        SQLGuardResult with is_safe=True if query is safe to execute,
        or is_safe=False with error message if not.
    """
    if not sql or not sql.strip():
        return SQLGuardResult(is_safe=False, sql=sql, error="Empty SQL query")

    try:
        # Parse SQL using sqlglot
        parsed = sqlglot.parse(sql, dialect="duckdb")

        if not parsed:
            return SQLGuardResult(
                is_safe=False, sql=sql, error="Failed to parse SQL"
            )

        for statement in parsed:
            if statement is None:
                continue

            # Check for forbidden statement types
            for forbidden in FORBIDDEN_STATEMENTS:
                if isinstance(statement, forbidden):
                    return SQLGuardResult(
                        is_safe=False,
                        sql=sql,
                        error=f"Forbidden SQL operation: {forbidden.__name__}",
                    )

            # Only allow SELECT statements (including UNION, INTERSECT, EXCEPT)
            if not isinstance(statement, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
                stmt_type = type(statement).__name__
                return SQLGuardResult(
                    is_safe=False,
                    sql=sql,
                    error=f"Only SELECT statements allowed, got: {stmt_type}",
                )

            # Check for forbidden functions
            for func in statement.find_all(exp.Anonymous, exp.Func):
                func_name = func.name.lower() if hasattr(func, "name") else ""
                if func_name in FORBIDDEN_FUNCTIONS:
                    return SQLGuardResult(
                        is_safe=False,
                        sql=sql,
                        error=f"Forbidden function: {func_name}",
                    )

            # Check for forbidden table-valued functions (e.g., read_csv, read_json)
            for forbidden_tvf in FORBIDDEN_TABLE_FUNCTIONS:
                if statement.find(forbidden_tvf):
                    return SQLGuardResult(
                        is_safe=False,
                        sql=sql,
                        error=f"Forbidden function: {forbidden_tvf.__name__}",
                    )

        # Add LIMIT if not present
        sql_with_limit = _ensure_limit(sql, parsed)

        logger.debug(f"[SQLGuard] Query validated: {sql[:50]}...")
        return SQLGuardResult(is_safe=True, sql=sql_with_limit)

    except sqlglot.errors.ParseError as e:
        return SQLGuardResult(
            is_safe=False, sql=sql, error=f"SQL parse error: {e}"
        )
    except Exception as e:
        logger.error(f"[SQLGuard] Validation error: {e}")
        return SQLGuardResult(
            is_safe=False, sql=sql, error=f"Validation error: {e}"
        )


def _ensure_limit(sql: str, parsed: list) -> str:
    """Ensure SQL has a LIMIT clause to prevent unbounded queries."""
    if not parsed:
        return sql

    statement = parsed[0]
    if not isinstance(statement, exp.Select):
        return sql

    # Check if LIMIT already exists
    if statement.args.get("limit"):
        return sql

    # Add LIMIT clause
    try:
        limited = statement.limit(MAX_ROWS)
        return limited.sql(dialect="duckdb")
    except Exception:
        # Fall back to string append if transformation fails
        return f"{sql.rstrip().rstrip(';')} LIMIT {MAX_ROWS}"


__all__ = ["validate_sql", "SQLGuardResult", "MAX_ROWS"]
