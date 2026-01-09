"""
Intent handlers for fetch node dispatch.

Each handler fetches CRM data for a specific intent type.
Follows Open/Closed principle - add new intents without modifying fetch node.

Usage:
    from backend.agent.fetch.tools import dispatch_intent, IntentContext, IntentResult
"""

from backend.agent.fetch.tools.activity import (
    handle_activities,
    handle_analytics,
    handle_fallback,
    tool_analytics,
    # Tools
    tool_recent_activity,
    tool_recent_history,
    tool_search_activities,
)
from backend.agent.fetch.tools.common import (
    IntentContext,
    IntentResult,
    ToolResult,
    make_sources,
    tool_company_lookup,
    with_datastore,
)
from backend.agent.fetch.tools.company import (
    handle_attachments,
    handle_company_search,
    handle_company_status,
    handle_contacts,
    tool_accounts_needing_attention,
    # Tools
    tool_search_attachments,
    tool_search_companies,
    tool_search_contacts,
)
from backend.agent.fetch.tools.pipeline import (
    handle_deals_at_risk,
    handle_forecast,
    handle_forecast_accuracy,
    handle_pipeline_summary,
    handle_renewals,
    tool_deals_at_risk,
    tool_forecast,
    tool_forecast_accuracy,
    # Tools
    tool_pipeline,
    tool_pipeline_by_owner,
    tool_pipeline_summary,
    tool_upcoming_renewals,
)

# Intent dispatcher - maps intent strings to handler functions
# 11 intents → 11 handlers (1:1 mapping)
INTENT_HANDLERS = {
    # Company-specific (triggers RAG in parallel)
    "company": handle_company_status,
    # Aggregate/global queries
    "pipeline_summary": handle_pipeline_summary,
    "renewals": handle_renewals,
    "deals_at_risk": handle_deals_at_risk,
    "forecast": handle_forecast,
    "forecast_accuracy": handle_forecast_accuracy,
    "activities": handle_activities,
    "contacts": handle_contacts,
    "company_search": handle_company_search,
    "attachments": handle_attachments,
    "analytics": handle_analytics,
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
    if ctx.resolved_company_id and intent not in _GLOBAL_INTENTS:
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
