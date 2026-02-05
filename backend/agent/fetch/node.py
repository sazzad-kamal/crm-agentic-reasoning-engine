"""Fetch node - plans and executes SQL queries."""

import logging
from typing import Any, cast

from backend.act_fetch import DEMO_MODE, act_fetch
from backend.agent.fetch.planner import SQLPlan, get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def _create_progress_collector() -> tuple[list[dict[str, str]], Any]:
    """Create a list and callback for collecting progress events.

    Returns:
        Tuple of (progress_list, callback_function)
    """
    progress_events: list[dict[str, str]] = []

    def on_progress(step: str, status: str) -> None:
        progress_events.append({"step": step, "status": status})

    return progress_events, on_progress


def _execute_sql_with_retry(
    sql_plan: SQLPlan,
    question: str,
    history: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute SQL with single retry on failure."""
    if not sql_plan.sql:
        logger.info("[Fetch] No SQL to execute")
        return [], None

    try:
        conn = get_connection()
        rows, error = execute_sql(sql_plan.sql, conn)

        # Retry with error feedback if query failed
        if error:
            logger.info(f"[Fetch] Retrying failed query: {error}")
            retry_plan = get_sql_plan(
                question=question,
                conversation_history=history,
                previous_error=error,
            )
            if retry_plan.sql:
                rows, error = execute_sql(retry_plan.sql, conn)
            if not error:
                logger.info("[Fetch] Retry succeeded")

        logger.info(f"[Fetch] SQL complete: {len(rows)} rows")
        return rows, error

    except Exception as e:
        logger.error(f"[Fetch] SQL execution failed: {e}")
        return [], str(e)


def fetch_node(state: AgentState) -> AgentState:
    """Unified fetch node that plans SQL and executes queries."""
    question = state["question"]

    # Demo mode: fetch from Act! API instead of SQL
    if DEMO_MODE:
        logger.info(f"[Fetch] Demo mode - calling Act! API for: {question[:50]}...")

        # Create progress collector for progressive loading UI
        progress_events, on_progress = _create_progress_collector()

        act_result = act_fetch(question, on_progress=on_progress)
        if act_result.get("error"):
            return cast(AgentState, {
                "sql_results": {},
                "fetch_progress": progress_events,
                "error": act_result["error"],
            })
        sql_results = act_result["data"]
        # Include cache timestamp if data came from cache
        if act_result.get("_cached_at"):
            sql_results["_cached_at"] = act_result["_cached_at"]
        return cast(AgentState, {
            "sql_results": sql_results,
            "fetch_progress": progress_events,
        })

    history = format_conversation_for_prompt(state.get("messages", []))
    logger.info(f"[Fetch] Processing: {question[:50]}...")

    # Initialize result state
    result: dict[str, Any] = {
        "sql_results": {},
    }

    # Step 1: Plan SQL
    sql_plan: SQLPlan | None = None
    try:
        sql_plan = get_sql_plan(question=question, conversation_history=history)
        logger.info(f"[Fetch] SQL planned: {sql_plan.sql[:60]}...")
    except Exception as e:
        logger.error(f"[Fetch] SQL planning failed: {e}")
        result["error"] = f"Query planning failed: {e}"
        return cast(AgentState, result)

    # Step 2: Execute SQL
    rows, error = _execute_sql_with_retry(sql_plan, question, history)
    if rows:
        result["sql_results"] = {"data": rows}
    if error:
        result["error"] = f"SQL execution failed: {error}"

    return cast(AgentState, result)


__all__ = ["fetch_node"]
