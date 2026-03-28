"""Compare node - handles comparison queries like 'Compare Q1 vs Q2 revenue'."""

import logging
import re
from typing import Any, cast

from backend.agent.fetch.planner import get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import safe_execute
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


# Patterns for identifying comparison elements
COMPARISON_PATTERNS = [
    r"compare\s+(.+?)\s+(?:vs\.?|versus|to|with|and)\s+(.+)",
    r"(.+?)\s+(?:vs\.?|versus)\s+(.+)",
    r"difference\s+between\s+(.+?)\s+and\s+(.+)",
    r"(.+?)\s+compared\s+to\s+(.+)",
]

# Time period mappings
QUARTER_MONTHS = {
    "q1": ("01", "03"),
    "q2": ("04", "06"),
    "q3": ("07", "09"),
    "q4": ("10", "12"),
}


def _extract_comparison_entities(question: str) -> tuple[str | None, str | None]:
    """Extract the two entities being compared from the question."""
    q_lower = question.lower()

    for pattern in COMPARISON_PATTERNS:
        match = re.search(pattern, q_lower, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()

    return None, None


def _is_time_period(entity: str) -> bool:
    """Check if entity represents a time period."""
    time_indicators = [
        "q1", "q2", "q3", "q4", "quarter",
        "month", "year", "week",
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
        "2023", "2024", "2025", "2026",
        "last", "this", "previous", "current",
    ]
    entity_lower = entity.lower()
    return any(indicator in entity_lower for indicator in time_indicators)


def _generate_comparison_sql(entity: str, question: str, history: str) -> str | None:
    """Generate SQL for one side of the comparison."""
    # Create a focused question for this entity
    focused_question = f"Get data for {entity} related to: {question}"

    try:
        sql_plan = get_sql_plan(
            question=focused_question,
            conversation_history=history,
        )
        return sql_plan.sql
    except Exception as e:
        logger.error(f"[Compare] SQL planning failed for {entity}: {e}")
        return None


def _execute_comparison_sql(sql: str) -> tuple[list[dict[str, Any]], str | None]:
    """Validate and execute comparison SQL."""
    if not sql:
        return [], None
    try:
        return safe_execute(sql, get_connection())
    except Exception as e:
        logger.error(f"[Compare] SQL execution failed: {e}")
        return [], str(e)


def _calculate_comparison(
    data_a: list[dict[str, Any]],
    data_b: list[dict[str, Any]],
    entity_a: str,
    entity_b: str,
) -> dict[str, Any]:
    """Calculate comparison metrics between two datasets."""
    comparison = {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "data_a": data_a,
        "data_b": data_b,
        "metrics": {},
    }

    # Find numeric columns to compare
    numeric_cols: set[str] = set()
    for row in data_a + data_b:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                numeric_cols.add(key)

    # Calculate aggregates and differences
    for col in numeric_cols:
        sum_a = sum(row.get(col, 0) or 0 for row in data_a)
        sum_b = sum(row.get(col, 0) or 0 for row in data_b)

        diff = sum_b - sum_a
        pct_change = ((sum_b - sum_a) / sum_a * 100) if sum_a != 0 else 0

        comparison["metrics"][col] = {
            f"{entity_a}": sum_a,
            f"{entity_b}": sum_b,
            "difference": diff,
            "percent_change": round(pct_change, 2),
        }

    # Add row counts
    comparison["metrics"]["_row_count"] = {
        f"{entity_a}": len(data_a),
        f"{entity_b}": len(data_b),
        "difference": len(data_b) - len(data_a),
    }

    return comparison


def compare_node(state: AgentState) -> AgentState:
    """Compare node that handles A vs B comparison queries."""
    question = state["question"]
    history = format_conversation_for_prompt(state.get("messages", []))
    logger.info(f"[Compare] Processing: {question[:50]}...")

    result: dict[str, Any] = {
        "sql_results": {},
    }

    # Step 1: Extract comparison entities
    entity_a, entity_b = _extract_comparison_entities(question)

    if not entity_a or not entity_b:
        logger.warning("[Compare] Could not extract comparison entities")
        result["error"] = "Could not identify what to compare. Please specify like 'Compare X vs Y'."
        return cast(AgentState, result)

    logger.info(f"[Compare] Comparing: '{entity_a}' vs '{entity_b}'")

    # Step 2: Generate and execute SQL for each entity
    sql_a = _generate_comparison_sql(entity_a, question, history)
    sql_b = _generate_comparison_sql(entity_b, question, history)

    data_a, error_a = _execute_comparison_sql(sql_a)
    data_b, error_b = _execute_comparison_sql(sql_b)

    # Step 3: Calculate comparison metrics
    comparison = _calculate_comparison(data_a, data_b, entity_a, entity_b)

    # Build result
    sql_results: dict[str, Any] = {
        "_debug": {
            "sql_a": sql_a,
            "sql_b": sql_b,
            "entity_a": entity_a,
            "entity_b": entity_b,
        },
        "comparison": comparison,
    }

    # Include raw data if available
    if data_a or data_b:
        sql_results["data"] = {
            entity_a: data_a,
            entity_b: data_b,
        }

    result["sql_results"] = sql_results

    if error_a:
        result["error"] = f"Error fetching {entity_a}: {error_a}"
    elif error_b:
        result["error"] = f"Error fetching {entity_b}: {error_b}"

    return cast(AgentState, result)


__all__ = ["compare_node"]
