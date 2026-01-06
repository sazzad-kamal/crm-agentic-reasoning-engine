"""
Intent handlers for fetch node dispatch.

Each handler fetches CRM data for a specific intent type.
Follows Open/Closed principle - add new intents without modifying fetch node.

Usage:
    from backend.agent.fetch.handlers import dispatch_intent, IntentContext, IntentResult
"""

from backend.agent.fetch.handlers.common import (
    IntentContext,
    IntentResult,
    make_sources,
    with_datastore,
    ToolResult,
)
from backend.agent.fetch.handlers.company import (
    handle_company_status,
    handle_company_search,
    handle_contacts,
    handle_attachments,
    # Tools
    tool_company_lookup,
    tool_search_companies,
    tool_search_contacts,
    tool_search_attachments,
    tool_accounts_needing_attention,
)
from backend.agent.fetch.handlers.pipeline import (
    handle_pipeline_summary,
    handle_renewals,
    handle_deals_at_risk,
    handle_forecast,
    handle_forecast_accuracy,
    # Tools
    tool_pipeline,
    tool_pipeline_summary,
    tool_pipeline_by_owner,
    tool_upcoming_renewals,
    tool_deals_at_risk,
    tool_forecast,
    tool_forecast_accuracy,
)
from backend.agent.fetch.handlers.activity import (
    handle_activities,
    handle_analytics,
    handle_fallback,
    # Tools
    tool_recent_activity,
    tool_recent_history,
    tool_search_activities,
    tool_analytics,
)


# Intent dispatcher - maps intent strings to handler functions
# Explicit mappings for all router intents (no implicit fallthrough)
INTENT_HANDLERS = {
    # Aggregate queries (no company_id required)
    "pipeline_summary": handle_pipeline_summary,
    "renewals": handle_renewals,
    "deals_at_risk": handle_deals_at_risk,  # At-risk/stalled deals
    "forecast": handle_forecast,  # Pipeline projections
    "forecast_accuracy": handle_forecast_accuracy,  # Win rate metrics
    "activities": handle_activities,
    "company_search": handle_company_search,
    "attachments": handle_attachments,
    "analytics": handle_analytics,  # Counts, breakdowns, aggregations
    # Contact queries
    "contact_lookup": handle_contacts,
    "contact_search": handle_contacts,
    # Company-specific queries (all route to handle_company_status)
    "company_status": handle_company_status,
    "pipeline": handle_company_status,
    "history": handle_company_status,  # Explicit: was implicit fallthrough
    "account_context": handle_company_status,  # Explicit: triggers Account RAG in parallel node
    "general": handle_company_status,  # Explicit: fallback for ambiguous queries
}


# Intents that always use their dedicated handler (global aggregates)
_GLOBAL_INTENTS = {
    "pipeline_summary",
    "deals_at_risk",
    "forecast",
    "forecast_accuracy",
    "company_search",
    "analytics",
}


def dispatch_intent(intent: str, ctx: IntentContext) -> IntentResult:
    """
    Dispatch to the appropriate intent handler.

    Simple dict lookup with one override: company-specific queries get full context.
    """
    # If there's a company and intent isn't a global aggregate, fetch full company context
    has_company = ctx.resolved_company_id or (
        ctx.router_result and getattr(ctx.router_result, "company_name_query", None)
    )
    if has_company and intent not in _GLOBAL_INTENTS:
        return handle_company_status(ctx)

    # Simple dict lookup with fallback
    return INTENT_HANDLERS.get(intent, handle_fallback)(ctx)


__all__ = [
    # Core
    "IntentContext",
    "IntentResult",
    "dispatch_intent",
    "INTENT_HANDLERS",
    # Helpers
    "make_sources",
    "with_datastore",
    "ToolResult",
    # Company tools
    "tool_company_lookup",
    "tool_search_companies",
    "tool_search_contacts",
    "tool_search_attachments",
    "tool_accounts_needing_attention",
    # Pipeline tools
    "tool_pipeline",
    "tool_pipeline_summary",
    "tool_pipeline_by_owner",
    "tool_upcoming_renewals",
    "tool_deals_at_risk",
    "tool_forecast",
    "tool_forecast_accuracy",
    # Activity tools
    "tool_recent_activity",
    "tool_recent_history",
    "tool_search_activities",
    "tool_analytics",
]
