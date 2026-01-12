"""
Slot-based query system for reliable SQL generation.

Instead of LLM generating raw SQL, it outputs structured slots:
- table: which table to query
- filters: WHERE conditions
- columns: SELECT columns (optional)
- order_by: ORDER BY clause (optional)

We then build valid SQL programmatically - no syntax errors possible.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============================================================================
# Table Definitions
# =============================================================================

TableName = Literal[
    "opportunities",
    "contacts",
    "activities",
    "companies",
    "history",
    "attachments",
]

# SQL columns per table (excludes RAG fields: notes, description)
# RAG fields come from vector search, not SQL
TABLE_COLUMNS: dict[TableName, list[str]] = {
    "opportunities": ["opportunity_id", "company_id", "name", "stage", "value", "owner", "close_date", "type"],
    "contacts": ["contact_id", "company_id", "first_name", "last_name", "email", "phone", "job_title", "role"],
    "activities": ["activity_id", "company_id", "contact_id", "opportunity_id", "type", "subject", "date"],
    "companies": ["company_id", "name", "status", "plan", "account_owner", "health_flags", "renewal_date"],
    "history": ["history_id", "company_id", "contact_id", "type", "date"],
    "attachments": ["attachment_id", "company_id", "name", "type", "url"],
}


# =============================================================================
# Pydantic Models
# =============================================================================


class SlotQuery(BaseModel):
    """A single slot-based query."""

    table: TableName = Field(description="Which table to query")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Filter conditions as column: value pairs",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Columns to select. None means SELECT *",
    )
    order_by: str | None = Field(
        default=None,
        description="ORDER BY clause (e.g., 'value DESC')",
    )
    purpose: str = Field(
        default="data",
        description="What this query fetches (for logging)",
    )


class SlotPlan(BaseModel):
    """LLM output containing slot-based queries."""

    queries: list[SlotQuery] = Field(
        default_factory=list,
        description="List of slot queries to execute",
    )
    needs_rag: bool = Field(
        default=False,
        description="Whether RAG context is needed (for 'why', 'what happened', discussion questions)",
    )


# =============================================================================
# SQL Builder
# =============================================================================


def _escape_value(value: Any) -> str:
    """Escape a value for SQL."""
    if isinstance(value, str):
        # Escape single quotes
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    elif isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    elif isinstance(value, (int, float)):
        return str(value)
    elif value is None:
        return "NULL"
    else:
        return f"'{value}'"


def _build_condition(key: str, value: Any) -> str:
    """Build a single WHERE condition."""
    # Handle special filter keys
    if key == "value_gt":
        return f"value > {value}"
    elif key == "value_lt":
        return f"value < {value}"
    elif key == "date_after":
        return f"date > {_escape_value(value)}"
    elif key == "stage_not_in":
        if isinstance(value, list):
            escaped = ", ".join(_escape_value(v) for v in value)
            return f"stage NOT IN ({escaped})"
        return f"stage != {_escape_value(value)}"
    elif key == "name" and isinstance(value, str):
        # For company name, use ILIKE for case-insensitive partial match
        return f"name ILIKE {_escape_value(f'%{value}%')}"
    elif isinstance(value, list):
        # IN clause for list values
        escaped = ", ".join(_escape_value(v) for v in value)
        return f"{key} IN ({escaped})"
    else:
        return f"{key} = {_escape_value(value)}"


def build_sql(slot: SlotQuery) -> str:
    """
    Build SQL query from a SlotQuery.

    This function constructs valid SQL programmatically,
    eliminating syntax errors from LLM-generated SQL.
    """
    # SELECT clause - use explicit columns to exclude RAG fields
    cols = ", ".join(slot.columns) if slot.columns else ", ".join(TABLE_COLUMNS[slot.table])
    sql = f"SELECT {cols} FROM {slot.table}"

    # WHERE clause - skip null/empty values
    if slot.filters:
        # Filter out None values (from strict schema nullable fields)
        active_filters = {k: v for k, v in slot.filters.items() if v is not None}
        if active_filters:
            conditions = [_build_condition(k, v) for k, v in active_filters.items()]
            sql += " WHERE " + " AND ".join(conditions)

    # ORDER BY clause
    if slot.order_by:
        sql += f" ORDER BY {slot.order_by}"

    logger.debug("Built SQL for '%s': %s", slot.purpose, sql)
    return sql


def build_sql_with_company_join(slot: SlotQuery) -> str:
    """
    Build SQL with JOIN to companies table for company name filtering.

    Used when filtering by company_name on tables that have company_id.
    """
    table = slot.table
    # Filter out None values first
    filters = {k: v for k, v in slot.filters.items() if v is not None}

    # Check if we need company name join
    company_name = filters.pop("company_name", None)

    # SELECT clause - use explicit columns to exclude RAG fields
    if slot.columns:
        cols = ", ".join(f"{table}.{c}" for c in slot.columns)
    else:
        table_cols = ", ".join(f"{table}.{c}" for c in TABLE_COLUMNS[table])
        cols = f"{table_cols}, companies.name as company_name"

    if company_name:
        sql = f"""SELECT {cols}
FROM {table}
JOIN companies ON {table}.company_id = companies.company_id
WHERE companies.name ILIKE {_escape_value(f'%{company_name}%')}"""

        # Add remaining filters
        for k, v in filters.items():
            sql += f" AND {_build_condition(k, v)}"
    else:
        sql = f"SELECT {cols} FROM {table}"
        if filters:
            conditions = [_build_condition(k, v) for k, v in filters.items()]
            sql += " WHERE " + " AND ".join(conditions)

    # ORDER BY clause
    if slot.order_by:
        sql += f" ORDER BY {slot.order_by}"

    logger.debug("Built SQL with join for '%s': %s", slot.purpose, sql)
    return sql


def slot_to_sql(slot: SlotQuery) -> str:
    """
    Convert a SlotQuery to SQL, handling company name joins automatically.
    """
    # Check if company_name is present AND not None
    company_name = slot.filters.get("company_name")
    if company_name is not None and slot.table != "companies":
        return build_sql_with_company_join(slot)
    return build_sql(slot)


__all__ = ["SlotQuery", "SlotPlan", "slot_to_sql"]
