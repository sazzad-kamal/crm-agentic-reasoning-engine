"""SQL query executor - executes SQL queries against DuckDB."""

import logging
from typing import TYPE_CHECKING, Any

import duckdb

if TYPE_CHECKING:
    from backend.agent.fetch.planner import SQLPlan

logger = logging.getLogger(__name__)


def execute_sql(
    plan: "SQLPlan",
    conn: duckdb.DuckDBPyConnection,
    max_rows: int = 100,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute SQL plan against DuckDB.

    Returns:
        Tuple of (rows, error_msg)
    """
    if not plan.sql or not plan.sql.strip():
        return [], None

    sql = plan.sql.strip()
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
