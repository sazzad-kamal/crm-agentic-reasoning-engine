"""CRM data fetch node using SQL executor."""

import logging
import time

from backend.agent.core.state import AgentState, Source
from backend.agent.datastore.connection import get_connection
from backend.agent.fetch.executor import execute_query_plan
from backend.agent.route.query_planner import QueryPlan, get_query_plan

logger = logging.getLogger(__name__)

# Maximum retry attempts for SQL execution failures
MAX_SQL_RETRIES = 1


def fetch_crm_node(state: AgentState) -> AgentState:
    """
    Execute SQL queries from query_plan and return results.

    Sets sql_results and resolved_company_id in state for downstream nodes.
    """
    start_time = time.time()

    query_plan: QueryPlan | None = state.get("query_plan")

    # If no query plan (error in route_node), return empty results
    if not query_plan or not query_plan.queries:
        logger.info("[FetchCRM] No queries to execute")
        return {
            "sql_results": {},
            "raw_data": {},
        }

    logger.info(f"[FetchCRM] Executing {len(query_plan.queries)} SQL queries...")

    try:
        # Get DuckDB connection with CSV tables loaded
        conn = get_connection()

        # Execute all queries in the plan
        sql_results, resolved, stats = execute_query_plan(query_plan, conn)

        # Retry with error feedback if any queries failed
        if stats.failed > 0 and stats.errors:
            error_summary = stats.get_error_summary()
            question = state.get("question", "")
            owner = state.get("owner")
            conversation_history = state.get("conversation_history", "")

            logger.info(f"[FetchCRM] Retrying {stats.failed} failed queries with error feedback")

            # Get new query plan with error feedback
            retry_plan = get_query_plan(
                question=question,
                conversation_history=conversation_history,
                owner=owner,
                error_feedback=error_summary,
            )

            # Execute retry plan
            retry_results, retry_resolved, retry_stats = execute_query_plan(retry_plan, conn)

            # Merge results - retry results override empty results from first attempt
            for purpose, data in retry_results.items():
                if data:  # Only override if retry got data
                    sql_results[purpose] = data
                    stats.success += 1
                    if purpose in stats.errors:
                        del stats.errors[purpose]

            # Merge resolved IDs
            resolved.update(retry_resolved)

            logger.info(f"[FetchCRM] Retry recovered {retry_stats.success}/{retry_stats.total} queries")

        # Extract resolved entity IDs for RAG filtering
        resolved_company_id = resolved.get("$company_id")
        resolved_contact_id = resolved.get("$contact_id")
        resolved_opportunity_id = resolved.get("$opportunity_id")

        # Build sources from results
        sources = _build_sources_from_results(sql_results)

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[FetchCRM] Complete in {latency_ms}ms, "
            f"results={list(sql_results.keys())}, "
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
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[FetchCRM] Failed after {latency_ms}ms: {e}")
        return {
            "sql_results": {},
            "raw_data": {},
            "error": f"SQL execution failed: {e}",
            "sql_queries_total": len(query_plan.queries) if query_plan else 0,
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


__all__ = ["fetch_crm_node"]
