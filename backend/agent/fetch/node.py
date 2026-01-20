"""
Unified fetch node - combines SQL planning, execution, and RAG retrieval.

This is the single data-fetching node in the workflow that:
1. Plans SQL query from user question
2. Executes SQL against DuckDB
3. Retrieves RAG context if needed
"""

import logging
from typing import Any, cast

from backend.agent.fetch.planner import SQLPlan, get_sql_plan
from backend.agent.fetch.rag.search import fetch_rag_context
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def _execute_sql_with_retry(
    sql_plan: SQLPlan,
    question: str,
    history: str,
) -> tuple[list[dict[str, Any]], dict[str, str], str | None]:
    """Execute SQL plan with retry on failure.

    Returns:
        Tuple of (rows, entity_ids, error_msg)
    """
    if not sql_plan.sql:
        logger.info("[Fetch] No SQL to execute")
        return [], {}, None

    try:
        conn = get_connection()
        rows, ids, error = execute_sql(sql_plan, conn)

        # Retry with error feedback if query failed
        if error:
            logger.info(f"[Fetch] Retrying failed query: {error}")
            retry_plan = get_sql_plan(
                question=question,
                conversation_history=history,
                previous_error=error,
            )
            retry_rows, retry_ids, retry_error = execute_sql(retry_plan, conn)

            if not retry_error:
                rows, ids, error = retry_rows, retry_ids, retry_error
                logger.info("[Fetch] Retry succeeded")

        logger.info(f"[Fetch] SQL complete: {len(rows)} rows, ids={list(ids.keys())}")
        return rows, ids, error

    except Exception as e:
        logger.error(f"[Fetch] SQL execution failed: {e}")
        return [], {}, str(e)


def fetch_node(state: AgentState) -> AgentState:
    """Unified fetch node that plans SQL, executes queries, and retrieves RAG context."""
    question = state["question"]
    history = format_conversation_for_prompt(state.get("messages", []))
    logger.info(f"[Fetch] Processing: {question[:50]}...")

    # Initialize result state
    result: dict[str, Any] = {
        "sql_results": {},
        "rag_context": "",
    }

    # Step 1: Plan SQL
    sql_plan: SQLPlan | None = None
    try:
        sql_plan = get_sql_plan(question=question, conversation_history=history)
        logger.info(f"[Fetch] SQL planned: {sql_plan.sql[:60]}..., needs_rag={sql_plan.needs_rag}")
    except Exception as e:
        logger.error(f"[Fetch] SQL planning failed: {e}")
        result["error"] = f"Query planning failed: {e}"
        return cast(AgentState, result)

    # Step 2: Execute SQL
    rows, entity_ids, error = _execute_sql_with_retry(sql_plan, question, history)
    if rows:
        result["sql_results"] = {"data": rows}
    if error:
        result["error"] = f"SQL execution failed: {error}"

    # Step 3: Fetch RAG using resolved IDs
    if sql_plan.needs_rag:
        context = fetch_rag_context(question, entity_ids)
        if context:
            result["rag_context"] = context
    else:
        logger.info("[Fetch] RAG skipped (needs_rag=False)")

    return cast(AgentState, result)


__all__ = ["fetch_node"]
