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
) -> tuple[list[dict[str, Any]], dict[str, str], str | None]:
    """
    Execute SQL plan against DuckDB.

    Returns:
        Tuple of (rows, entity_ids, error_msg)
        - rows: list of dicts with query results
        - entity_ids: dict of resolved IDs for RAG filtering
        - error_msg: error message if failed, None if success
    """
    if not plan.sql or not plan.sql.strip():
        return [], {}, None

    sql = plan.sql.strip()
    if "LIMIT" not in sql.upper():
        sql = f"{sql} LIMIT {max_rows}"

    try:
        result = conn.execute(sql)
        cols = [d[0] for d in result.description]
        rows = [dict(zip(cols, r, strict=True)) for r in result.fetchall()]

        # Extract entity IDs for RAG filtering
        ids: dict[str, str] = {}
        if rows:
            for key in ("company_id", "contact_id", "opportunity_id"):
                if val := rows[0].get(key):
                    ids[key] = str(val)

        return rows, ids, None

    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return [], {}, str(e)


__all__ = ["execute_sql"]
