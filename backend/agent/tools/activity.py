"""
Activity, history, and analytics tools.
"""

from backend.agent.datastore import CRMDataStore, get_datastore
from backend.agent.schemas import Source, ToolResult
from backend.agent.tools.base import make_sources, with_datastore


@with_datastore
def tool_recent_activity(
    company_id: str,
    days: int = 90,
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get recent activities for a company."""
    ds = datastore

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
            activities, "activities", company_id,
            f"Activities for {company_name} (last {days} days)"
        )
    )


@with_datastore
def tool_recent_history(
    company_id: str,
    days: int = 90,
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get recent history entries (calls, emails, notes) for a company."""
    ds = datastore

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
            history, "history", company_id,
            f"History for {company_name} (last {days} days)"
        )
    )


@with_datastore
def tool_search_activities(
    activity_type: str = "",
    days: int = 30,
    company_id: str = "",
    limit: int = 30,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Search activities by type, date range, or company."""
    ds = datastore

    activities = ds.search_activities(
        activity_type=activity_type, days=days,
        company_id=company_id, limit=limit
    )

    search_desc = []
    if activity_type: search_desc.append(f"type='{activity_type}'")
    if company_id: search_desc.append(f"company='{company_id}'")
    search_desc.append(f"last {days} days")

    label = f"Activities: {', '.join(search_desc)}"

    return ToolResult(
        data={
            "count": len(activities),
            "activities": activities,
            "filters": {"activity_type": activity_type, "days": days, "company_id": company_id},
        },
        sources=make_sources(activities, "activities", "search", label)
    )


@with_datastore
def tool_analytics(
    metric: str,
    group_by: str = "",
    company_id: str = "",
    group_id: str = "",
    activity_type: str = "",
    days: int = 30,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Get analytics and breakdowns for CRM data.

    Supports metrics: contact_breakdown, activity_breakdown, activity_count,
    accounts_by_group, pipeline_by_group.
    """
    ds = datastore

    match metric:
        case "contact_breakdown":
            resolved_company = ds.resolve_company_id(company_id) if company_id else None
            data = ds.get_contact_breakdown(company_id=resolved_company, group_by=group_by or "role")

            company_name = ""
            if resolved_company:
                company = ds.get_company(resolved_company)
                company_name = company.get("name", resolved_company) if company else resolved_company

            label = f"Contact breakdown by {group_by or 'role'}"
            if company_name: label += f" for {company_name}"

            return ToolResult(
                data=data,
                sources=[Source(type="analytics", id=f"contacts_{resolved_company or 'all'}", label=label)]
            )

        case "activity_breakdown":
            resolved_company = ds.resolve_company_id(company_id) if company_id else None
            data = ds.get_activity_breakdown(company_id=resolved_company, days=days, group_by=group_by or "type")

            company_name = ""
            if resolved_company:
                company = ds.get_company(resolved_company)
                company_name = company.get("name", resolved_company) if company else resolved_company

            label = f"Activity breakdown by {group_by or 'type'} (last {days} days)"
            if company_name: label += f" for {company_name}"

            return ToolResult(
                data=data,
                sources=[Source(type="analytics", id=f"activities_{resolved_company or 'all'}", label=label)]
            )

        case "activity_count":
            resolved_company = ds.resolve_company_id(company_id) if company_id else None
            data = ds.get_activity_count_by_filter(
                activity_type=activity_type or None, days=days, company_id=resolved_company
            )

            company_name = ""
            if resolved_company:
                company = ds.get_company(resolved_company)
                company_name = company.get("name", resolved_company) if company else resolved_company

            label_parts = [f"Activity count (last {days} days)"]
            if activity_type: label_parts.append(f"type={activity_type}")
            if company_name: label_parts.append(f"for {company_name}")

            return ToolResult(
                data=data,
                sources=[Source(type="analytics", id="activity_count", label=" ".join(label_parts))]
            )

        case "accounts_by_group":
            data = ds.get_accounts_by_group()
            return ToolResult(
                data=data,
                sources=[Source(type="analytics", id="accounts_by_group", label="Account distribution by group")]
            )

        case "pipeline_by_group":
            data = ds.get_pipeline_by_group(group_id=group_id or None)
            label = f"Pipeline for group: {group_id}" if group_id else "Pipeline by group"

            return ToolResult(
                data=data,
                sources=[Source(type="analytics", id=f"pipeline_{group_id or 'all_groups'}", label=label)]
            )

        case _:
            return ToolResult(
                data={
                    "error": f"Unknown metric: {metric}",
                    "available_metrics": [
                        "contact_breakdown", "activity_breakdown", "activity_count",
                        "accounts_by_group", "pipeline_by_group",
                    ]
                },
                sources=[],
                error=f"Unknown analytics metric: {metric}"
            )


__all__ = [
    "tool_recent_activity",
    "tool_recent_history",
    "tool_search_activities",
    "tool_analytics",
]
