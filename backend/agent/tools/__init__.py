"""
CRM Tools for the agentic layer.

Each tool returns both data and source citations.
Tools are organized by domain:
- company.py: Company, contact, group, and attachment lookups
- pipeline.py: Pipeline, renewals, and forecast tools
- activity.py: Activity, history, and analytics tools
"""

from backend.agent.schemas import ToolResult
from backend.agent.tools.base import make_sources

# Company and contact tools
from backend.agent.tools.company import (
    tool_company_lookup,
    tool_search_companies,
    tool_contact_lookup,
    tool_search_contacts,
    tool_group_members,
    tool_list_groups,
    tool_search_attachments,
    tool_accounts_needing_attention,
)

# Pipeline tools
from backend.agent.tools.pipeline import (
    tool_pipeline,
    tool_pipeline_summary,
    tool_pipeline_by_owner,
    tool_upcoming_renewals,
    tool_deals_at_risk,
    tool_forecast,
    tool_forecast_accuracy,
)

# Activity tools
from backend.agent.tools.activity import (
    tool_recent_activity,
    tool_recent_history,
    tool_search_activities,
    tool_analytics,
)


__all__ = [
    "ToolResult",
    "make_sources",
    # Company tools
    "tool_company_lookup",
    "tool_search_companies",
    "tool_contact_lookup",
    "tool_search_contacts",
    "tool_group_members",
    "tool_list_groups",
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
