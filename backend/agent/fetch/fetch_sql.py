"""SQL data fetch node."""

import logging

from backend.agent.core.state import AgentState, Source, format_history_for_prompt
from backend.agent.datastore.connection import get_connection
from backend.agent.fetch.executor import execute_sql_plan
from backend.agent.route.sql_planner import SQLPlan, get_sql_plan

logger = logging.getLogger(__name__)


def fetch_sql_node(state: AgentState) -> AgentState:
    """
    Execute SQL query from sql_plan and return results.

    Sets sql_results and resolved_company_id in state for downstream nodes.
    """
    sql_plan: SQLPlan | None = state.get("sql_plan")

    # If no SQL plan (error in route_node), return empty results
    if not sql_plan or not sql_plan.sql:
        logger.info("[FetchSQL] No SQL to execute")
        return {
            "sql_results": {},
            "raw_data": {},
        }

    logger.info(f"[FetchSQL] Executing SQL: {sql_plan.sql[:60]}...")

    try:
        # Get DuckDB connection with CSV tables loaded
        conn = get_connection()

        # Execute the SQL query
        sql_results, resolved, stats = execute_sql_plan(sql_plan, conn)

        # Retry with error feedback if query failed
        if stats.failed > 0 and stats.errors:
            logger.info("[FetchSQL] Retrying failed query with error feedback")

            retry_plan = get_sql_plan(
                question=state.get("question", ""),
                conversation_history=f"{format_history_for_prompt(state.get('messages', []))}\n\n[PREVIOUS QUERY FAILED]\n{stats.get_error_summary()}\nPlease fix the query.",
            )

            # Execute retry plan
            retry_results, retry_resolved, retry_stats = execute_sql_plan(retry_plan, conn)

            # Use retry results if successful
            if retry_stats.success > 0:
                sql_results = retry_results
                resolved = retry_resolved
                stats = retry_stats

            logger.info(f"[FetchSQL] Retry: {retry_stats.success}/{retry_stats.total}")

        # Extract resolved entity IDs for RAG filtering
        resolved_company_id = resolved.get("$company_id")
        resolved_contact_id = resolved.get("$contact_id")
        resolved_opportunity_id = resolved.get("$opportunity_id")

        # Build sources from results
        sources = _build_sources_from_results(sql_results)

        logger.info(
            f"[FetchSQL] Complete: results={list(sql_results.keys())}, "
            f"resolved={{company={resolved_company_id}, contact={resolved_contact_id}, opp={resolved_opportunity_id}}}, "
            f"sql_success={stats.success}/{stats.total}"
        )

        return {
            "sql_results": sql_results,
            "resolved_company_id": resolved_company_id,
            "resolved_contact_id": resolved_contact_id,
            "resolved_opportunity_id": resolved_opportunity_id,
            "raw_data": sql_results,  # Alias for UI/followup compatibility
            "sources": sources,
            "sql_queries_total": stats.total,
            "sql_queries_success": stats.success,
        }

    except Exception as e:
        logger.error(f"[FetchSQL] Failed: {e}")
        return {
            "sql_results": {},
            "raw_data": {},
            "error": f"SQL execution failed: {e}",
            "sql_queries_total": 1,
            "sql_queries_success": 0,
        }


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


__all__ = ["fetch_sql_node"]
