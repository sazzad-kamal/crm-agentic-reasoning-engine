"""
Fetch nodes - sequential data retrieval for LangGraph workflow.

Exports:
    fetch_sql_node: Execute SQL queries from query plan
    fetch_rag_node: Fetch account context via RAG
    execute_query_plan: Execute SQL queries against DuckDB
"""

from backend.agent.fetch.executor import (
    SQLExecutionError,
    SQLValidationError,
    execute_query_plan,
)
from backend.agent.fetch.fetch_rag import fetch_rag_node
from backend.agent.fetch.fetch_sql import fetch_sql_node

__all__ = [
    "fetch_sql_node",
    "fetch_rag_node",
    "execute_query_plan",
    "SQLExecutionError",
    "SQLValidationError",
]
