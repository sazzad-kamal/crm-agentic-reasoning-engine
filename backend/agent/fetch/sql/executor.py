"""SQL query executor - executes SQL queries against DuckDB."""

import logging
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


def execute_sql(
    sql: str,
    conn: duckdb.DuckDBPyConnection,
    max_rows: int = 100,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute SQL query against DuckDB.

    Args:
        sql: SQL query string (caller should validate non-empty)
        conn: DuckDB connection
        max_rows: Maximum rows to return

    Returns:
        Tuple of (rows, error_msg)
    """
    sql = sql.strip()
    if "LIMIT" not in sql.upper():
        sql = f"{sql} LIMIT {max_rows}"

    try:
        result = conn.execute(sql)
        cols = [d[0] for d in result.description]
        rows = [dict(zip(cols, r, strict=True)) for r in result.fetchall()]
        return rows, None

    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return [], str(e)


__all__ = ["execute_sql"]
