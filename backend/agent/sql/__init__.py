"""Shared SQL infrastructure for all agent nodes.

Provides DuckDB connection, query planning, execution,
safety validation, and schema management.
"""

from backend.agent.sql.connection import get_connection
from backend.agent.sql.executor import execute_sql, safe_execute
from backend.agent.sql.guard import validate_sql
from backend.agent.sql.planner import SQLPlan, get_sql_plan

__all__ = ["execute_sql", "safe_execute", "get_connection", "validate_sql", "SQLPlan", "get_sql_plan"]
