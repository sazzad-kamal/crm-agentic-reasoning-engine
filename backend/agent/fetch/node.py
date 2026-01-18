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
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def _capture_eval_data(
    sql_plan: SQLPlan | None,
    rows: list[dict[str, Any]],
    error: str | None,
    account_rag_invoked: bool,
    account_chunks: list[str],
) -> None:
    """Capture eval-specific data via context variable (no-op if eval not running)."""
    try:
        from backend.eval.shared.callback import set_eval_data

        set_eval_data(
            sql_plan=sql_plan,
            sql_queries_total=1 if sql_plan and sql_plan.sql else 0,
            sql_queries_success=1 if rows and not error else 0,
            account_rag_invoked=account_rag_invoked,
            account_chunks=account_chunks,
        )
    except ImportError:
        pass


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
                conversation_history=f"{history}\n\n[PREVIOUS QUERY FAILED]\n{error}\nPlease fix the query.",
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


def _fetch_rag(
    question: str,
    entity_ids: dict[str, str],
) -> tuple[str, bool, list[str]]:
    """Fetch RAG context for resolved entities.

    Returns:
        Tuple of (context_str, rag_invoked, chunks) for eval capture.
    """
    # Filter to valid entity ID keys
    filters = {
        k: v
        for k, v in entity_ids.items()
        if k in ("company_id", "contact_id", "opportunity_id")
    }

    if not filters:
        logger.info("[Fetch] RAG skipped (no entity IDs resolved)")
        return "", False, []

    logger.info(f"[Fetch] Retrieving RAG context with filters={filters}")

    try:
        from backend.agent.fetch.rag.search import search_entity_context

        context, _ = search_entity_context(question, filters)
        chunks = context.split("\n\n---\n\n") if context else []

        logger.info("[Fetch] RAG complete")
        return context, True, chunks

    except Exception as e:
        logger.warning(f"RAG fetch failed: {e}")
        return "", False, []


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
        _capture_eval_data(None, [], str(e), False, [])
        return cast(AgentState, result)

    # Step 2: Execute SQL
    rows, entity_ids, error = _execute_sql_with_retry(sql_plan, question, history)
    if rows:
        result["sql_results"] = {"data": rows}
    if error:
        result["error"] = f"SQL execution failed: {error}"

    # Step 3: Fetch RAG using resolved IDs
    if sql_plan.needs_rag:
        context, rag_invoked, chunks = _fetch_rag(question, entity_ids)
        if context:
            result["rag_context"] = context
    else:
        logger.info("[Fetch] RAG skipped (needs_rag=False)")
        context, rag_invoked, chunks = "", False, []

    # Capture eval data
    _capture_eval_data(sql_plan, rows, error, rag_invoked, chunks)

    return cast(AgentState, result)


__all__ = ["fetch_node"]
