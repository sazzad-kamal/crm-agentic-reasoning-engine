"""Schemas for the route module."""

from pydantic import BaseModel


class RouterResult(BaseModel):
    """Result from the router's LLM analysis.

    Contains intent classification and extracted parameters.
    """

    company_id: str | None = None  # Resolved from LLM's company_name
    intent: str = "pipeline_summary"  # From LLM (15 intents)
    # Extracted parameters from LLM
    segment: str | None = None  # Enterprise, Mid-Market, SMB
    industry: str | None = None  # Software, Manufacturing, etc.
    role: str | None = None  # Decision Maker, Champion, Executive
    activity_type: str | None = None  # Call, Email, Meeting, Task
    analytics_metric: str | None = None  # contact_breakdown, activity_count, etc.
    analytics_group_by: str | None = None  # role, type, stage


__all__ = ["RouterResult"]
