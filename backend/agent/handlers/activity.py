"""
Activity-related intent handlers.

Handles activities, analytics, and fallback intents.
"""

from backend.agent.extractors import extract_activity_type
from backend.agent.tools.activity import tool_search_activities, tool_analytics
from backend.agent.tools.pipeline import tool_upcoming_renewals

from backend.agent.handlers.common import (
    IntentContext,
    IntentResult,
    empty_raw_data,
    apply_tool_result,
    safe_extend,
    logger,
)


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

    # Default based on query type
    if is_count_query:
        return "activity_count", "", ""
    if is_breakdown_query:
        return "activity_breakdown", "type", ""

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
    logger.debug("[Data] No specific intent, fetching general renewals")
    result = IntentResult(raw_data=empty_raw_data())

    renewals_result = tool_upcoming_renewals(days=ctx.days)
    apply_tool_result(result, renewals_result, "renewals_data", "renewals")
    return result


__all__ = [
    "handle_activities",
    "handle_analytics",
    "handle_fallback",
]
