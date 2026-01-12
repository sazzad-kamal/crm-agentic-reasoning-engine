"""
Slot-based query system for reliable SQL generation.

Instead of LLM generating raw SQL, it outputs structured slots:
- table: which table to query
- filters: list of {field, op, value} conditions
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
# Pydantic Models
# =============================================================================


class Filter(BaseModel):
    """A single filter condition."""

    field: str = Field(description="Column name to filter on")
    op: Literal["eq", "neq", "gt", "lt", "gte", "lte", "in", "not_in", "like"] = Field(
        description="Operator: eq, neq, gt, lt, gte, lte, in, not_in, like"
    )
    value: Any = Field(description="Value to compare against")


class SlotQuery(BaseModel):
    """A single slot-based query."""

    table: str = Field(description="Which table to query")
    filters: list[Filter] = Field(
        default_factory=list,
        description="List of filter conditions",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Columns to select. None means all columns",
    )
    order_by: str | None = Field(
        default=None,
        description="ORDER BY clause (e.g., 'value DESC')",
    )


class SlotPlan(BaseModel):
    """LLM output containing slot-based queries."""

    queries: list[SlotQuery] = Field(
        default_factory=list,
        description="List of slot queries to execute",
    )
    needs_rag: bool = Field(
        default=False,
        description="Whether RAG context is needed",
    )


# =============================================================================
# SQL Builder with pypika
# =============================================================================


def _build_criterion(table: Table, f: Filter) -> Any:
    """Build a pypika criterion from a Filter."""
    col = table[f.field]
    val = f.value

    if f.op == "eq":
        # Case-insensitive partial match for string fields like 'name'
        if f.field in ("name", "health_flags") and isinstance(val, str):
            return Lower(col).like(f"%{val.lower()}%")
        return col == val
    elif f.op == "neq":
        return col != val
    elif f.op == "gt":
        return col > val
    elif f.op == "lt":
        return col < val
    elif f.op == "gte":
        return col >= val
    elif f.op == "lte":
        return col <= val
    elif f.op == "in":
        return col.isin(val) if isinstance(val, list) else col == val
    elif f.op == "not_in":
        return col.notin(val) if isinstance(val, list) else col != val
    elif f.op == "like":
        pattern = f"%{val}%" if not val.startswith("%") else val
        return Lower(col).like(pattern.lower())
    else:
        raise ValueError(f"Unknown operator: {f.op}")


def _parse_order_by(order_by: str) -> tuple[str, Order]:
    """Parse 'field DESC' into (field, Order)."""
    parts = order_by.strip().split()
    field = parts[0]
    direction = Order.desc if len(parts) > 1 and parts[1].upper() == "DESC" else Order.asc
    return field, direction


def _build_query(slot: SlotQuery) -> Query:
    """Build a pypika Query from a SlotQuery."""
    table = Table(slot.table)

    # SELECT columns (use * if not specified)
    if slot.columns:
        query = Query.from_(table).select(*[table[c] for c in slot.columns])
    else:
        query = Query.from_(table).select("*")

    # WHERE filters
    for f in slot.filters:
        query = query.where(_build_criterion(table, f))

    # ORDER BY
    if slot.order_by:
        field, direction = _parse_order_by(slot.order_by)
        query = query.orderby(table[field], order=direction)

    return query


def slot_to_sql(slot: SlotQuery) -> str:
    """Convert a SlotQuery to SQL using pypika."""
    query = _build_query(slot)
    sql: str = query.get_sql()
    logger.debug("Built SQL for '%s': %s", slot.table, sql)
    return sql


__all__ = ["Filter", "SlotQuery", "SlotPlan", "slot_to_sql"]
