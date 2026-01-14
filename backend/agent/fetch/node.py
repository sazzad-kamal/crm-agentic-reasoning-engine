"""
Unified fetch node - combines SQL planning, execution, and RAG retrieval.

This is the single data-fetching node in the workflow that:
1. Plans SQL query from user question
2. Executes SQL against DuckDB
3. Retrieves RAG context if needed
"""

import logging
from typing import Any, cast

from backend.agent.core.state import AgentState, Source, format_history_for_prompt
from backend.agent.fetch.planner import SQLPlan, get_sql_plan
from backend.agent.fetch.sql.connection import get_connection
from backend.agent.fetch.sql.executor import execute_sql_plan

logger = logging.getLogger(__name__)


def _build_sources_from_results(
    sql_results: dict[str, list[dict]],
) -> list[Source]:
    """
    Build source references from SQL query results.

    Maps query purposes to source types for citations.
    """
    sources = []

    # Map purposes to source types
    purpose_to_type = {
        "company_info": "company",
        "companies": "company",
        "open_deals": "opportunities",
        "opportunities": "opportunities",
        "stale_deals": "opportunities",
        "weighted_forecast": "forecast",
        "contact_info": "contact",
        "contacts": "contacts",
        "interaction_history": "history",
        "history": "history",
        "activities": "activities",
        "renewals": "renewals",
        "attachments": "attachments",
    }

    for purpose, data in sql_results.items():
        if not data:
            continue

        source_type = purpose_to_type.get(purpose, purpose)

        # Use first row's ID if available
        first_row = data[0]
        source_id = (
            first_row.get("company_id")
            or first_row.get("contact_id")
            or first_row.get("opportunity_id")
            or purpose
        )

        # Create descriptive label
        label = f"{len(data)} {source_type}" if len(data) > 1 else source_type

        sources.append(
            Source(
                type=source_type,
                id=str(source_id),
                label=label,
            )
        )

    return sources


def _fetch_rag_context(  # pragma: no cover
    question: str,
    filters: dict[str, str],
) -> tuple[str, list[Source], list[str]]:
    """
    Fetch RAG context with error handling.

    Uses lazy import of tool_entity_rag. Covered via fetch_node integration tests.

    Args:
        question: The user's question
        filters: Dict of entity IDs to filter by

    Returns:
        Tuple of (context string, sources, chunks)
    """
    try:
        from backend.agent.fetch.rag.tools import tool_entity_rag

        context, sources = tool_entity_rag(question, filters)

        # Split combined context into individual chunks for RAGAS evaluation
        chunks = context.split("\n\n---\n\n") if context else []

        return context, sources, chunks

    except Exception as e:
        logger.warning(f"RAG fetch failed: {e}")
        return "", [], []


def fetch_node(state: AgentState) -> AgentState:
    """
    Unified fetch node that plans SQL, executes queries, and retrieves RAG context.

    This combines the previous route_node, fetch_sql_node, and fetch_rag_node
    into a single orchestrating node.
    """
    question = state["question"]
    logger.info(f"[Fetch] Processing: {question[:50]}...")

    # Initialize result state
    result: dict[str, Any] = {
        "sql_results": {},
        "raw_data": {},
        "account_context_answer": "",
        "account_rag_invoked": False,
    }

    # ----- Step 1: Plan SQL -----
    try:
        sql_plan = get_sql_plan(
            question=question,
            conversation_history=format_history_for_prompt(state.get("messages", [])),
        )
        result["sql_plan"] = sql_plan
        result["needs_rag"] = sql_plan.needs_rag
        logger.info(f"[Fetch] SQL planned: {sql_plan.sql[:60]}..., needs_rag={sql_plan.needs_rag}")

    except Exception as e:
        logger.error(f"[Fetch] SQL planning failed: {e}")
        result["sql_plan"] = SQLPlan(sql="", needs_rag=False)
        result["needs_rag"] = False
        result["error"] = f"Query planning failed: {e}"
        return cast(AgentState, result)

    # ----- Step 2: Execute SQL -----
    if sql_plan.sql:
        try:
            conn = get_connection()
            sql_results, resolved, stats = execute_sql_plan(sql_plan, conn)

            # Retry with error feedback if query failed
            if stats.failed > 0 and stats.errors:
                logger.info("[Fetch] Retrying failed query with error feedback")

                retry_plan = get_sql_plan(
                    question=question,
                    conversation_history=f"{format_history_for_prompt(state.get('messages', []))}\n\n[PREVIOUS QUERY FAILED]\n{stats.get_error_summary()}\nPlease fix the query.",
                )

                retry_results, retry_resolved, retry_stats = execute_sql_plan(retry_plan, conn)

                if retry_stats.success > 0:
                    sql_results = retry_results
                    resolved = retry_resolved
                    stats = retry_stats

                logger.info(f"[Fetch] Retry: {retry_stats.success}/{retry_stats.total}")

            # Extract resolved entity IDs
            resolved_company_id = resolved.get("$company_id")
            resolved_contact_id = resolved.get("$contact_id")
            resolved_opportunity_id = resolved.get("$opportunity_id")

            # Build sources
            sources = _build_sources_from_results(sql_results)

            result.update({
                "sql_results": sql_results,
                "resolved_company_id": resolved_company_id,
                "resolved_contact_id": resolved_contact_id,
                "resolved_opportunity_id": resolved_opportunity_id,
                "raw_data": sql_results,
                "sources": sources,
                "sql_queries_total": stats.total,
                "sql_queries_success": stats.success,
            })

            logger.info(
                f"[Fetch] SQL complete: results={list(sql_results.keys())}, "
                f"resolved={{company={resolved_company_id}, contact={resolved_contact_id}, opp={resolved_opportunity_id}}}"
            )

        except Exception as e:
            logger.error(f"[Fetch] SQL execution failed: {e}")
            result.update({
                "error": f"SQL execution failed: {e}",
                "sql_queries_total": 1,
                "sql_queries_success": 0,
            })
    else:
        logger.info("[Fetch] No SQL to execute")

    # ----- Step 3: Fetch RAG if needed -----
    if sql_plan.needs_rag:
        # Build filters from resolved entity IDs
        filters: dict[str, str] = {}
        company_id = result.get("resolved_company_id")
        contact_id = result.get("resolved_contact_id")
        opportunity_id = result.get("resolved_opportunity_id")

        if company_id:
            filters["company_id"] = company_id
        if contact_id:
            filters["contact_id"] = contact_id
        if opportunity_id:
            filters["opportunity_id"] = opportunity_id

        if filters:
            logger.info(f"[Fetch] Retrieving RAG context with filters={filters}")

            rag_context, rag_sources, rag_chunks = _fetch_rag_context(question, filters)

            result.update({
                "account_context_answer": rag_context,
                "sources": result.get("sources", []) + rag_sources,
                "account_chunks": rag_chunks,
                "account_rag_invoked": True,
            })

            logger.info(f"[Fetch] RAG complete: sources={len(rag_sources)}, chunks={len(rag_chunks)}")
        else:
            logger.info("[Fetch] RAG skipped (no entity IDs resolved)")
    else:
        logger.info("[Fetch] RAG skipped (needs_rag=False)")

    return cast(AgentState, result)


__all__ = ["fetch_node"]
