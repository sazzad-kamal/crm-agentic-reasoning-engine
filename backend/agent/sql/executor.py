"""SQL query executor - executes SQL queries against DuckDB."""

import logging
from typing import Any

import duckdb

from backend.agent.sql.guard import validate_sql

logger = logging.getLogger(__name__)


def execute_sql(
    sql: str,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute SQL and return (rows, error_msg)."""
    try:
        result = conn.execute(sql)
        cols = [d[0] for d in result.description]
        rows = [dict(zip(cols, r, strict=True)) for r in result.fetchall()]
        return rows, None

    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return [], str(e)


def safe_execute(sql: str, conn: duckdb.DuckDBPyConnection) -> tuple[list[dict[str, Any]], str | None]:
    """Validate SQL for safety, then execute. Returns (rows, error_msg)."""
    guard_result = validate_sql(sql)
    if not guard_result.is_safe:
        return [], f"Query blocked: {guard_result.error}"
    return execute_sql(guard_result.sql, conn)


__all__ = ["execute_sql", "safe_execute"]
