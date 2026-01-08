"""
Activity-related intent handlers and tools.

Handles activities, analytics, and fallback intents.
Includes tool functions merged from tools/activity.py.
"""

from backend.agent.fetch.handlers.common import (
    CRMDataStore,
    IntentContext,
    IntentResult,
    Source,
    ToolResult,
    apply_tool_result,
    empty_raw_data,
    logger,
    make_sources,
    safe_extend,
    with_datastore,
)
from backend.agent.fetch.handlers.extractors import extract_activity_type

# =============================================================================
# Tool Functions (merged from tools/activity.py)
# =============================================================================


@with_datastore
def tool_recent_activity(
    company_id: str, days: int = 90, limit: int = 20, datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get recent activities for a company."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore

    activities = ds.get_recent_activities(company_id, days=days, limit=limit)

    company = ds.get_company(company_id)
    company_name = company.get("name", company_id) if company else company_id

    return ToolResult(
        data={
            "company_id": company_id,
            "company_name": company_name,
            "days": days,
            "count": len(activities),
            "activities": activities,
        },
        sources=make_sources(
            activities,
            "activities",
            company_id,
            f"Activities for {company_name} (last {days} days)",
        ),
    )


@with_datastore
def tool_recent_history(
    company_id: str, days: int = 90, limit: int = 20, datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get recent history entries (calls, emails, notes) for a company."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore

    history = ds.get_recent_history(company_id, days=days, limit=limit)

    company = ds.get_company(company_id)
    company_name = company.get("name", company_id) if company else company_id

    return ToolResult(
        data={
            "company_id": company_id,
            "company_name": company_name,
            "days": days,
            "count": len(history),
            "history": history,
        },
        sources=make_sources(
            history, "history", company_id, f"History for {company_name} (last {days} days)"
        ),
    )


@with_datastore
def tool_search_activities(
    activity_type: str = "",
    days: int = 30,
    company_id: str = "",
    limit: int = 30,
    datastore: CRMDataStore | None = None,
) -> ToolResult:
    """Search activities by type, date range, or company."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore

    activities = ds.search_activities(
        activity_type=activity_type, days=days, company_id=company_id, limit=limit
    )

    search_desc = []
    if activity_type:
        search_desc.append(f"type='{activity_type}'")
    if company_id:
        search_desc.append(f"company='{company_id}'")
    search_desc.append(f"last {days} days")

    label = f"Activities: {', '.join(search_desc)}"

    return ToolResult(
        data={
            "count": len(activities),
            "activities": activities,
            "filters": {"activity_type": activity_type, "days": days, "company_id": company_id},
        },
        sources=make_sources(activities, "activities", "search", label),
    )


def _analytics_result(data: dict, source_id: str, label: str) -> ToolResult:
    """Helper to create analytics ToolResult."""
    return ToolResult(data=data, sources=[Source(type="analytics", id=source_id, label=label)])


@with_datastore
def tool_analytics(
    metric: str,
    group_by: str = "",
    company_id: str = "",
    group_id: str = "",
    activity_type: str = "",
    days: int = 30,
    datastore: CRMDataStore | None = None,
) -> ToolResult:
    """Get analytics and breakdowns for CRM data."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore
    # Resolve company ID and get company name
    resolved: str | None = None
    company_name = ""
    if company_id:
        resolved = ds.resolve_company_id(company_id)
        if resolved:
            company = ds.get_company(resolved)
            company_name = company.get("name", resolved) if company else resolved
    suffix = f" for {company_name}" if company_name else ""

    match metric:
        case "contact_breakdown":
            data = ds.get_contact_breakdown(company_id=resolved, group_by=group_by or "role")
            return _analytics_result(
                data,
                f"contacts_{resolved or 'all'}",
                f"Contact breakdown by {group_by or 'role'}{suffix}",
            )

        case "activity_breakdown":
            data = ds.get_activity_breakdown(
                company_id=resolved, days=days, group_by=group_by or "type"
            )
            return _analytics_result(
                data,
                f"activities_{resolved or 'all'}",
                f"Activity breakdown by {group_by or 'type'} (last {days} days){suffix}",
            )

        case "activity_count":
            data = ds.get_activity_count_by_filter(
                activity_type=activity_type or None, days=days, company_id=resolved
            )
            parts = [f"Activity count (last {days} days)"]
            if activity_type:
                parts.append(f"type={activity_type}")
            if company_name:
                parts.append(f"for {company_name}")
            return _analytics_result(data, "activity_count", " ".join(parts))

        case "accounts_by_group":
            return _analytics_result(
                ds.get_accounts_by_group(), "accounts_by_group", "Account distribution by group"
            )

        case "pipeline_by_group":
            data = ds.get_pipeline_by_group(group_id=group_id or None)
            label = f"Pipeline for group: {group_id}" if group_id else "Pipeline by group"
            return _analytics_result(data, f"pipeline_{group_id or 'all_groups'}", label)

        case _:
            return ToolResult(
                data={
                    "error": f"Unknown metric: {metric}",
                    "available_metrics": [
                        "contact_breakdown",
                        "activity_breakdown",
                        "activity_count",
                        "accounts_by_group",
                        "pipeline_by_group",
                    ],
                },
                sources=[],
                error=f"Unknown analytics metric: {metric}",
            )


# =============================================================================
# Intent Handlers
# =============================================================================


def handle_activities(ctx: IntentContext) -> IntentResult:
    """Handle activities intent (cross-company search)."""
    logger.debug("[Data] Searching activities across all companies")
    result = IntentResult(raw_data=empty_raw_data())

    activity_type = extract_activity_type(ctx.question)
    activities_result = tool_search_activities(activity_type=activity_type, days=ctx.days)
    apply_tool_result(result, activities_result, "activities_data", "activities", limit=10)
    return result


def _detect_analytics_metric(question: str) -> tuple[str, str, str]:
    """
    Detect the analytics metric type from the question.

    Uses general patterns, not exact question matching.

    Returns:
        (metric, group_by, activity_type) tuple
    """
    q = question.lower()

    # Pattern: counting/breakdown keywords
    is_count_query = any(w in q for w in ["how many", "count", "total", "number of"])
    is_breakdown_query = any(
        w in q for w in ["breakdown", "distribution", "percentage", "split", "ratio"]
    )
    is_comparison = any(w in q for w in ["most", "highest", "lowest", "compare", "common"])

    # Detect entity type
    has_contact = "contact" in q
    has_activity = "activit" in q

    # Detect specific activity types
    activity_type = ""
    for atype in ["email", "call", "meeting", "demo", "task"]:
        if atype in q:
            activity_type = atype
            break

    # Decision logic based on entities
    if has_contact and (is_breakdown_query or "role" in q):
        return "contact_breakdown", "role", ""

    if has_activity:
        if activity_type and is_count_query:
            return "activity_count", "", activity_type
        if is_breakdown_query or is_comparison or "type" in q:
            return "activity_breakdown", "type", ""
        if is_count_query:
            return "activity_count", "", ""

    # Default: activity breakdown by type
    return "activity_breakdown", "type", ""


def handle_analytics(ctx: IntentContext) -> IntentResult:
    """Handle analytics intent (counts, breakdowns, aggregations)."""
    logger.debug("[Data] Processing analytics query")
    result = IntentResult(raw_data=empty_raw_data(), resolved_company_id=ctx.resolved_company_id)

    metric, group_by, activity_type = _detect_analytics_metric(ctx.question)
    logger.debug(f"[Analytics] metric={metric}, group_by={group_by}, activity_type={activity_type}")

    analytics_result = tool_analytics(
        metric=metric,
        group_by=group_by,
        company_id=ctx.resolved_company_id or "",
        group_id="",
        activity_type=activity_type,
        days=ctx.days,
    )
    result.analytics_data = analytics_result.data
    safe_extend(result.sources, analytics_result.sources)
    result.raw_data["analytics"] = analytics_result.data
    return result


def handle_fallback(ctx: IntentContext) -> IntentResult:
    """Fallback handler for unknown intents."""
    # Import here to avoid circular imports
    from backend.agent.fetch.handlers.pipeline import tool_upcoming_renewals

    logger.debug("[Data] No specific intent, fetching general renewals")
    result = IntentResult(raw_data=empty_raw_data())

    renewals_result = tool_upcoming_renewals(days=ctx.days)
    apply_tool_result(result, renewals_result, "renewals_data", "renewals")
    return result


__all__ = [
    # Handlers
    "handle_activities",
    "handle_analytics",
    "handle_fallback",
    # Tools (for backward compatibility)
    "tool_recent_activity",
    "tool_recent_history",
    "tool_search_activities",
    "tool_analytics",
]
