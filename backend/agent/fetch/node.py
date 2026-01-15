"""
Unified fetch node - combines SQL planning, execution, and RAG retrieval.

This is the single data-fetching node in the workflow that:
1. Plans SQL query from user question
2. Executes SQL against DuckDB
3. Retrieves RAG context if needed
"""

import logging
from typing import Any, cast

from backend.agent.core.state import AgentState, format_conversation_for_prompt
from backend.agent.fetch.planner import SQLPlan, get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import SQLExecutionStats, execute_sql_plan

logger = logging.getLogger(__name__)


def _capture_eval_data(
    sql_plan: SQLPlan | None,
    stats: SQLExecutionStats | None,
    account_rag_invoked: bool,
    account_chunks: list[str],
) -> None:
    """Capture eval-specific data via context variable (no-op if eval not running)."""
    try:
        from backend.eval.callback import set_eval_data

        set_eval_data(
            sql_plan=sql_plan,
            sql_queries_total=stats.total if stats else 0,
            sql_queries_success=stats.success if stats else 0,
            account_rag_invoked=account_rag_invoked,
            account_chunks=account_chunks,
        )
    except ImportError:
        # Eval module not available (e.g., production without eval deps)
        pass

def _execute_sql_with_retry(
    sql_plan: SQLPlan,
    question: str,
    history: str,
) -> tuple[dict[str, Any], dict[str, str], SQLExecutionStats | None]:
    """Execute SQL plan with retry on failure.

    Returns:
        Tuple of (state_updates, resolved_ids, stats) where resolved_ids are internal.
    """
    if not sql_plan.sql:
        logger.info("[Fetch] No SQL to execute")
        return {}, {}, None

    try:
        conn = get_connection()
        sql_results, resolved, stats = execute_sql_plan(sql_plan, conn)

        # Retry with error feedback if query failed
        if stats.failed > 0 and stats.errors:
            logger.info("[Fetch] Retrying failed query with error feedback")
            retry_plan = get_sql_plan(
                question=question,
                conversation_history=f"{history}\n\n[PREVIOUS QUERY FAILED]\n{stats.get_error_summary()}\nPlease fix the query.",
            )
            retry_results, retry_resolved, retry_stats = execute_sql_plan(retry_plan, conn)

            if retry_stats.success > 0:
                sql_results, resolved, stats = retry_results, retry_resolved, retry_stats

            logger.info(f"[Fetch] Retry: {retry_stats.success}/{retry_stats.total}")

        logger.info(
            f"[Fetch] SQL complete: results={list(sql_results.keys())}, "
            f"resolved={{company={resolved.get('$company_id')}}}"
        )

        state_updates = {"sql_results": sql_results}

        # Convert $key to key for RAG filtering
        resolved_ids = {k.lstrip("$"): v for k, v in resolved.items() if v}

        return state_updates, resolved_ids, stats

    except Exception as e:
        logger.error(f"[Fetch] SQL execution failed: {e}")
        error_stats = SQLExecutionStats()
        error_stats.total = 1
        return {"error": f"SQL execution failed: {e}"}, {}, error_stats


def _fetch_rag_if_needed(
    needs_rag: bool,
    question: str,
    resolved_ids: dict[str, str],
) -> tuple[dict[str, Any], bool, list[str]]:
    """Fetch RAG context if needed.

    Returns:
        Tuple of (state_updates, rag_invoked, chunks) for eval capture.
    """
    if not needs_rag:
        logger.info("[Fetch] RAG skipped (needs_rag=False)")
        return {}, False, []

    # Filter to valid entity ID keys
    filters = {
        k: v
        for k, v in resolved_ids.items()
        if k in ("company_id", "contact_id", "opportunity_id", "activity_id")
    }

    if not filters:
        logger.info("[Fetch] RAG skipped (no entity IDs resolved)")
        return {}, False, []

    logger.info(f"[Fetch] Retrieving RAG context with filters={filters}")

    try:
        from backend.agent.fetch.rag.search import tool_entity_rag

        context, _ = tool_entity_rag(question, filters)
        chunks = context.split("\n\n---\n\n") if context else []

        logger.info("[Fetch] RAG complete")

        return {"account_context_answer": context}, True, chunks

    except Exception as e:
        logger.warning(f"RAG fetch failed: {e}")
        return {}, False, []


def fetch_node(state: AgentState) -> AgentState:
    """Unified fetch node that plans SQL, executes queries, and retrieves RAG context."""
    question = state["question"]
    history = format_conversation_for_prompt(state.get("messages", []))
    logger.info(f"[Fetch] Processing: {question[:50]}...")

    # Initialize result state
    result: dict[str, Any] = {
        "sql_results": {},
        "account_context_answer": "",
    }

    # Step 1: Plan SQL
    sql_plan: SQLPlan | None = None
    try:
        sql_plan = get_sql_plan(question=question, conversation_history=history)
        logger.info(f"[Fetch] SQL planned: {sql_plan.sql[:60]}..., needs_rag={sql_plan.needs_rag}")
    except Exception as e:
        logger.error(f"[Fetch] SQL planning failed: {e}")
        result["error"] = f"Query planning failed: {e}"
        _capture_eval_data(None, None, False, [])
        return cast(AgentState, result)

    # Step 2: Execute SQL (returns state updates + internal resolved IDs + stats)
    sql_updates, resolved_ids, stats = _execute_sql_with_retry(sql_plan, question, history)
    result.update(sql_updates)

    # Step 3: Fetch RAG using resolved IDs (internal to this node)
    rag_updates, rag_invoked, chunks = _fetch_rag_if_needed(sql_plan.needs_rag, question, resolved_ids)
    result.update(rag_updates)

    # Capture eval data out-of-band (no-op if eval not running)
    _capture_eval_data(sql_plan, stats, rag_invoked, chunks)

    return cast(AgentState, result)


__all__ = ["fetch_node"]
