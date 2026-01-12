"""SQL data fetch node."""

import logging

from backend.agent.core.state import AgentState, Source, format_history_for_prompt
from backend.agent.datastore.connection import get_connection
from backend.agent.fetch.executor import execute_slot_plan
from backend.agent.route.query_planner import SlotPlan, get_slot_plan

logger = logging.getLogger(__name__)


def fetch_sql_node(state: AgentState) -> AgentState:
    """
    Execute SQL queries from slot_plan and return results.

    Sets sql_results and resolved_company_id in state for downstream nodes.
    """
    slot_plan: SlotPlan | None = state.get("slot_plan")

    # If no slot plan (error in route_node), return empty results
    if not slot_plan or not slot_plan.queries:
        logger.info("[FetchSQL] No queries to execute")
        return {
            "sql_results": {},
            "raw_data": {},
        }

    logger.info(f"[FetchSQL] Executing {len(slot_plan.queries)} SQL queries...")

    try:
        # Get DuckDB connection with CSV tables loaded
        conn = get_connection()

        # Execute all queries in the plan
        sql_results, resolved, stats = execute_slot_plan(slot_plan, conn)

        # Retry with error feedback if any queries failed
        if stats.failed > 0 and stats.errors:
            error_summary = stats.get_error_summary()
            question = state.get("question", "")
            owner = state.get("owner")
            conversation_history = format_history_for_prompt(state.get("messages", []))

            logger.info(f"[FetchSQL] Retrying {stats.failed} failed queries with error feedback")

            # Get new slot plan with error feedback
            history = f"{conversation_history}\n\n[PREVIOUS QUERY FAILED]\n{error_summary}\nPlease fix the query."
            retry_plan = get_slot_plan(
                question=question,
                conversation_history=history,
                owner=owner,
            )

            # Execute retry plan
            retry_results, retry_resolved, retry_stats = execute_slot_plan(retry_plan, conn)

            # Merge results - retry results override empty results from first attempt
            for purpose, data in retry_results.items():
                if data:  # Only override if retry got data
                    sql_results[purpose] = data
                    stats.success += 1
                    if purpose in stats.errors:
                        del stats.errors[purpose]

            # Merge resolved IDs
            resolved.update(retry_resolved)

            logger.info(f"[FetchSQL] Retry recovered {retry_stats.success}/{retry_stats.total} queries")

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
            "sql_queries_total": len(slot_plan.queries) if slot_plan else 0,
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
