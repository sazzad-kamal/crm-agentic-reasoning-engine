"""
Fetch nodes - sequential data retrieval for LangGraph workflow.

Exports:
    fetch_crm_node: Execute SQL queries from query plan
    fetch_account_node: Fetch account context via RAG
    execute_query_plan: Execute SQL queries against DuckDB
"""

from backend.agent.fetch.executor import (
    SQLExecutionError,
    SQLValidationError,
    execute_query_plan,
)
from backend.agent.fetch.fetch_crm import fetch_crm_node
from backend.agent.fetch.fetch_entity import fetch_account_node

__all__ = [
    "fetch_crm_node",
    "fetch_account_node",
    "execute_query_plan",
    "SQLExecutionError",
    "SQLValidationError",
]
