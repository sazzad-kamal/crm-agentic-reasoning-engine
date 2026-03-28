"""Trend node - handles time-series analysis queries."""

import logging
import re
from typing import Any, cast

from backend.agent.fetch.planner import get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import safe_execute
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


# Time period keywords that indicate trend queries
TREND_KEYWORDS = [
    "trend", "over time", "growth", "historical",
    "month over month", "year over year", "yoy", "mom",
    "trajectory", "progression", "evolution",
    "by month", "by quarter", "by year", "by week",
    "monthly", "quarterly", "yearly", "weekly",
]

# Time granularity patterns
GRANULARITY_PATTERNS = {
    "daily": r"(?:daily|by day|per day)",
    "weekly": r"(?:weekly|by week|per week)",
    "monthly": r"(?:monthly|by month|per month|month over month|mom)",
    "quarterly": r"(?:quarterly|by quarter|per quarter)",
    "yearly": r"(?:yearly|annual|annually|by year|per year|year over year|yoy)",
}


def _detect_granularity(question: str) -> str:
    """Detect the time granularity from the question."""
    q_lower = question.lower()

    for granularity, pattern in GRANULARITY_PATTERNS.items():
        if re.search(pattern, q_lower):
            return granularity

    # Default to monthly for general trend questions
    return "monthly"


def _enhance_question_for_trend(question: str, granularity: str) -> str:
    """Enhance the question to ensure time-grouped results."""
    # Add explicit grouping instruction if not present
    group_terms = ["group by", "grouped by", "by month", "by quarter", "by year"]
    q_lower = question.lower()

    if not any(term in q_lower for term in group_terms):
        time_col_hint = {
            "daily": "DATE(created_at) or close_date",
            "weekly": "DATE_TRUNC('week', created_at) or close_date",
            "monthly": "DATE_TRUNC('month', created_at) or close_date",
            "quarterly": "DATE_TRUNC('quarter', created_at) or close_date",
            "yearly": "DATE_TRUNC('year', created_at) or close_date",
        }
        return f"{question}. Group results by {granularity} using {time_col_hint.get(granularity, 'date')}. Order by date ascending."

    return question


def _calculate_trend_metrics(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate trend analysis metrics from time-series data."""
    if len(data) < 2:
        return {"error": "Not enough data points for trend analysis"}

    metrics: dict[str, Any] = {
        "data_points": len(data),
        "columns": {},
    }

    # Find numeric columns for trend analysis
    numeric_cols: set[str] = set()
    for row in data:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                numeric_cols.add(key)

    # Calculate metrics for each numeric column
    for col in numeric_cols:
        values = [row.get(col, 0) or 0 for row in data]

        if not values or all(v == 0 for v in values):
            continue

        first_val = values[0] if values[0] != 0 else 1
        last_val = values[-1]
        total_change = last_val - values[0]
        pct_change = (total_change / first_val * 100) if first_val != 0 else 0

        # Calculate period-over-period changes
        period_changes = []
        for i in range(1, len(values)):
            if values[i - 1] != 0:
                change = (values[i] - values[i - 1]) / values[i - 1] * 100
                period_changes.append(round(change, 2))
            else:
                period_changes.append(0)

        # Determine trend direction
        if total_change > 0:
            direction = "increasing"
        elif total_change < 0:
            direction = "decreasing"
        else:
            direction = "stable"

        # Calculate average and volatility
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        volatility = variance ** 0.5

        metrics["columns"][col] = {
            "first_value": values[0],
            "last_value": last_val,
            "min_value": min(values),
            "max_value": max(values),
            "average": round(avg, 2),
            "total_change": round(total_change, 2),
            "percent_change": round(pct_change, 2),
            "direction": direction,
            "volatility": round(volatility, 2),
            "period_changes": period_changes,
        }

    return metrics


def trend_node(state: AgentState) -> AgentState:
    """Trend node that handles time-series analysis queries."""
    question = state["question"]
    history = format_conversation_for_prompt(state.get("messages", []))
    logger.info(f"[Trend] Processing: {question[:50]}...")

    result: dict[str, Any] = {
        "sql_results": {},
    }

    # Step 1: Detect granularity
    granularity = _detect_granularity(question)
    logger.info(f"[Trend] Detected granularity: {granularity}")

    # Step 2: Enhance question for trend analysis
    enhanced_question = _enhance_question_for_trend(question, granularity)

    # Step 3: Generate SQL
    try:
        sql_plan = get_sql_plan(
            question=enhanced_question,
            conversation_history=history,
        )
        logger.info(f"[Trend] SQL planned: {sql_plan.sql[:60] if sql_plan.sql else 'None'}...")
    except Exception as e:
        logger.error(f"[Trend] SQL planning failed: {e}")
        result["error"] = f"Trend query planning failed: {e}"
        return cast(AgentState, result)

    # Step 4: Execute SQL
    if not sql_plan.sql:
        result["error"] = "Could not generate trend analysis query"
        return cast(AgentState, result)

    try:
        rows, error = safe_execute(sql_plan.sql, get_connection())
        if error:
            result["error"] = f"SQL execution failed: {error}"
            return cast(AgentState, result)
    except Exception as e:
        logger.error(f"[Trend] SQL execution failed: {e}")
        result["error"] = str(e)
        return cast(AgentState, result)

    # Step 5: Calculate trend metrics
    trend_metrics = _calculate_trend_metrics(rows)

    # Build result
    sql_results: dict[str, Any] = {
        "_debug": {
            "sql": sql_plan.sql,
            "granularity": granularity,
            "row_count": len(rows),
        },
        "data": rows,
        "trend_analysis": trend_metrics,
    }

    result["sql_results"] = sql_results
    logger.info(f"[Trend] Analysis complete: {len(rows)} data points")

    return cast(AgentState, result)


__all__ = ["trend_node"]
