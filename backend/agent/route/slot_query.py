"""
Slot-based query system for reliable SQL generation.

Instead of LLM generating raw SQL, it outputs structured slots:
- table: which table to query
- filters: WHERE conditions
- columns: SELECT columns (optional)
- order_by: ORDER BY clause (optional)

We then build valid SQL programmatically using pypika.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from pypika import Order, Query, Table
from pypika.functions import Lower

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
]

# SQL columns per table (excludes RAG fields: notes, description)
# RAG fields come from vector search, not SQL
# Columns match the actual CSV schema in backend/data/csv/
TABLE_COLUMNS: dict[TableName, list[str]] = {
    "opportunities": ["opportunity_id", "company_id", "name", "stage", "value", "owner", "expected_close_date", "type"],
    "contacts": ["contact_id", "company_id", "first_name", "last_name", "email", "phone", "job_title", "role"],
    "activities": ["activity_id", "company_id", "contact_id", "opportunity_id", "type", "subject", "due_datetime"],
    "companies": ["company_id", "name", "status", "plan", "account_owner", "health_flags", "renewal_date"],
    "history": ["history_id", "company_id", "contact_id", "type", "date"],
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
# SQL Builder with pypika
# =============================================================================


def _build_criterion(table: Table, key: str, value: Any) -> Any:
    """Build a pypika criterion from a filter key-value pair."""
    # Handle special filter keys
    if key == "value_gt":
        return table.value > value
    elif key == "value_lt":
        return table.value < value
    elif key == "date_after":
        return table.date > value
    elif key == "stage_not_in":
        if isinstance(value, list):
            return table.stage.notin(value)
        return table.stage != value
    elif key == "name" and isinstance(value, str):
        # Case-insensitive partial match using LOWER
        return Lower(table.name).like(f"%{value.lower()}%")
    elif key == "health_flags" and isinstance(value, str):
        # Partial match for health flags (e.g., "at-risk" matches "at-risk-low-activity")
        return Lower(table.health_flags).like(f"%{value.lower()}%")
    elif isinstance(value, list):
        return table[key].isin(value)
    else:
        return table[key] == value


def _parse_order_by(order_by: str) -> tuple[str, Order]:
    """Parse 'field DESC' into (field, Order)."""
    parts = order_by.strip().split()
    field = parts[0]
    direction = Order.desc if len(parts) > 1 and parts[1].upper() == "DESC" else Order.asc
    return field, direction


def _build_query(slot: SlotQuery) -> Query:
    """Build a pypika Query from a SlotQuery."""
    table = Table(slot.table)

    # SELECT columns
    cols = slot.columns or TABLE_COLUMNS[slot.table]
    query = Query.from_(table).select(*[table[c] for c in cols])

    # WHERE filters - skip None values
    for key, value in slot.filters.items():
        if value is not None:
            query = query.where(_build_criterion(table, key, value))

    # ORDER BY
    if slot.order_by:
        field, direction = _parse_order_by(slot.order_by)
        query = query.orderby(table[field], order=direction)

    return query


def _build_query_with_company_join(slot: SlotQuery) -> Query:
    """Build a pypika Query with JOIN to companies table."""
    table = Table(slot.table)
    companies = Table("companies")

    # Get company_name filter
    filters = {k: v for k, v in slot.filters.items() if v is not None}
    company_name = filters.pop("company_name", None)

    # SELECT columns - include company name
    if slot.columns:
        cols = [table[c] for c in slot.columns]
    else:
        cols = [table[c] for c in TABLE_COLUMNS[slot.table]]
        cols.append(companies.name.as_("company_name"))

    # Build query with JOIN
    query = (
        Query.from_(table)
        .join(companies)
        .on(table.company_id == companies.company_id)
        .select(*cols)
    )

    # Add company name filter (case-insensitive partial match)
    if company_name:
        query = query.where(Lower(companies.name).like(f"%{company_name.lower()}%"))

    # Add remaining filters
    for key, value in filters.items():
        query = query.where(_build_criterion(table, key, value))

    # ORDER BY
    if slot.order_by:
        field, direction = _parse_order_by(slot.order_by)
        query = query.orderby(table[field], order=direction)

    return query


def slot_to_sql(slot: SlotQuery) -> str:
    """
    Convert a SlotQuery to SQL using pypika.

    Handles company name joins automatically when filtering by company_name
    on tables that have company_id.
    """
    # Check if company_name filter is present and not on companies table
    company_name = slot.filters.get("company_name")
    if company_name is not None and slot.table != "companies":
        query = _build_query_with_company_join(slot)
    else:
        query = _build_query(slot)

    sql: str = query.get_sql()
    logger.debug("Built SQL for '%s': %s", slot.purpose, sql)
    return sql


__all__ = ["SlotQuery", "SlotPlan", "slot_to_sql", "TABLE_COLUMNS", "TableName"]
