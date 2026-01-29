"""Entity context builder for data-aware follow-up generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import duckdb

from backend.agent.fetch.sql.executor import execute_sql

logger = logging.getLogger(__name__)

# Entity ID columns to look for in sql_results rows
_ENTITY_COLUMNS = ("company_id", "contact_id", "opportunity_id")

# Linked record count queries per entity type
_LINKED_COUNTS: dict[str, list[tuple[str, str]]] = {
    "company_id": [
        ("contacts", "SELECT COUNT(*) AS cnt FROM contacts WHERE company_id = '{id}'"),
        ("opportunities", "SELECT COUNT(*) AS cnt FROM opportunities WHERE company_id = '{id}'"),
        ("activities", "SELECT COUNT(*) AS cnt FROM activities WHERE company_id = '{id}'"),
        ("history", "SELECT COUNT(*) AS cnt FROM history WHERE company_id = '{id}'"),
    ],
    "contact_id": [
        ("activities", "SELECT COUNT(*) AS cnt FROM activities WHERE contact_id = '{id}'"),
        ("history", "SELECT COUNT(*) AS cnt FROM history WHERE contact_id = '{id}'"),
    ],
    "opportunity_id": [
        ("activities", "SELECT COUNT(*) AS cnt FROM activities WHERE opportunity_id = '{id}'"),
        ("history", "SELECT COUNT(*) AS cnt FROM history WHERE opportunity_id = '{id}'"),
    ],
}

# Name lookup queries per entity type
_NAME_QUERIES: dict[str, str] = {
    "company_id": "SELECT name FROM companies WHERE company_id = '{id}'",
    "contact_id": "SELECT first_name || ' ' || last_name AS name FROM contacts WHERE contact_id = '{id}'",
    "opportunity_id": "SELECT name FROM opportunities WHERE opportunity_id = '{id}'",
}

_ENTITY_LABELS: dict[str, str] = {
    "company_id": "company",
    "contact_id": "contact",
    "opportunity_id": "opportunity",
}


def _extract_entity_ids(sql_results: dict[str, Any]) -> dict[str, set[str]]:
    """Extract unique entity IDs from sql_results rows."""
    entities: dict[str, set[str]] = {col: set() for col in _ENTITY_COLUMNS}
    rows = sql_results.get("data", [])
    for row in rows:
        for col in _ENTITY_COLUMNS:
            val = row.get(col)
            if val:
                entities[col].add(str(val))
    return {col: ids for col, ids in entities.items() if ids}


def _get_entity_name(
    col: str, entity_id: str, conn: duckdb.DuckDBPyConnection
) -> str:
    """Look up entity name. Returns ID as fallback."""
    query = _NAME_QUERIES.get(col)
    if not query:
        return entity_id
    rows, error = execute_sql(query.format(id=entity_id), conn)
    if error or not rows:
        return entity_id
    name: str = rows[0].get("name", entity_id)
    return name


def _get_linked_counts(
    col: str, entity_id: str, conn: duckdb.DuckDBPyConnection
) -> list[str]:
    """Get linked record count strings for an entity."""
    counts = []
    for table, query in _LINKED_COUNTS.get(col, []):
        rows, error = execute_sql(query.format(id=entity_id), conn)
        if error or not rows:
            continue
        cnt = rows[0].get("cnt", 0)
        if cnt > 0:
            counts.append(f"{cnt} {table}")
    return counts


def get_entity_context(
    sql_results: dict[str, Any],
    conn: duckdb.DuckDBPyConnection,
) -> str:
    """Build entity context string from sql_results for follow-up prompt.

    Returns compact summary of linked data for entities found in the answer,
    or empty string if no entities found.
    """
    entities = _extract_entity_ids(sql_results)
    if not entities:
        return ""

    lines: list[str] = []
    for col, ids in entities.items():
        label = _ENTITY_LABELS[col]
        for entity_id in sorted(ids):
            name = _get_entity_name(col, entity_id, conn)
            counts = _get_linked_counts(col, entity_id, conn)
            if counts:
                lines.append(f"- {name} ({label}): {', '.join(counts)}")
            else:
                lines.append(f"- {name} ({label}): no linked records")

    if not lines:
        return ""

    return "Linked data for entities in the answer:\n" + "\n".join(lines)


__all__ = ["get_entity_context"]
