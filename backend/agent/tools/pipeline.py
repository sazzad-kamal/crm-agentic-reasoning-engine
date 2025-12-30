"""
Pipeline, renewals, and forecast tools.
"""

from backend.agent.datastore import CRMDataStore, get_datastore
from backend.agent.schemas import Source, ToolResult
from backend.agent.tools.base import make_sources, with_datastore


@with_datastore
def tool_pipeline(
    company_id: str,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get pipeline summary and open opportunities for a company."""
    ds = datastore

    summary = ds.get_pipeline_summary(company_id)
    opportunities = ds.get_open_opportunities(company_id, limit=20)

    company = ds.get_company(company_id)
    company_name = company.get("name", company_id) if company else company_id

    has_pipeline = opportunities or summary.get("total_count", 0) > 0

    return ToolResult(
        data={
            "company_id": company_id,
            "company_name": company_name,
            "summary": summary,
            "opportunities": opportunities,
        },
        sources=make_sources(has_pipeline, "opportunities", company_id, f"Pipeline for {company_name}")
    )


@with_datastore
def tool_pipeline_summary(datastore: CRMDataStore | None = None) -> ToolResult:
    """Get overall pipeline summary across all companies."""
    ds = datastore
    summary = ds.get_all_pipeline_summary()

    return ToolResult(
        data=summary,
        sources=[Source(
            type="pipeline", id="all",
            label=f"Pipeline: {summary['total_count']} deals, ${summary['total_value']:,.0f}"
        )] if summary.get("total_count", 0) > 0 else []
    )


@with_datastore
def tool_pipeline_by_owner(
    owner: str = "",
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get pipeline summary grouped by owner."""
    ds = datastore
    data = ds.get_pipeline_by_owner(owner=owner or None)

    label = f"Pipeline for {owner}" if owner else "Pipeline by owner"

    return ToolResult(
        data=data,
        sources=[Source(type="pipeline", id=f"by_owner_{owner or 'all'}", label=label)]
        if data.get("total_count", 0) > 0 else []
    )


@with_datastore
def tool_upcoming_renewals(
    days: int = 90,
    limit: int = 20,
    owner: str = "",
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get companies with upcoming renewals."""
    ds = datastore
    renewals = ds.get_upcoming_renewals(days=days, limit=limit, owner=owner or None)

    label = f"Renewals for {owner} (next {days} days)" if owner else f"Upcoming renewals (next {days} days)"

    return ToolResult(
        data={
            "days": days,
            "count": len(renewals),
            "owner_filter": owner or None,
            "renewals": renewals,
        },
        sources=make_sources(renewals, "renewals", "upcoming", label)
    )


@with_datastore
def tool_deals_at_risk(
    owner: str = "",
    days_threshold: int = 45,
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get deals that are at risk (stale, need attention)."""
    ds = datastore

    deals = ds.get_deals_at_risk(owner=owner or None, days_threshold=days_threshold, limit=limit)

    label = f"At-risk deals for {owner}" if owner else f"At-risk deals (>{days_threshold} days in stage)"

    return ToolResult(
        data={
            "count": len(deals),
            "days_threshold": days_threshold,
            "owner_filter": owner or None,
            "deals": deals,
        },
        sources=make_sources(deals, "opportunities", "at_risk", label)
    )


@with_datastore
def tool_forecast(
    owner: str = "",
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get weighted pipeline forecast."""
    ds = datastore
    data = ds.get_forecast(owner=owner or None)

    label = f"Forecast for {owner}" if owner else "Pipeline forecast"

    return ToolResult(
        data=data,
        sources=[Source(
            type="forecast",
            id=f"forecast_{owner or 'all'}",
            label=f"{label}: ${data.get('total_weighted', 0):,.0f} weighted"
        )] if data.get("total_pipeline", 0) > 0 else []
    )


@with_datastore
def tool_forecast_accuracy(
    owner: str = "",
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get forecast accuracy metrics (win rate) based on historical closed deals."""
    ds = datastore
    data = ds.get_forecast_accuracy(owner=owner or None)

    label = f"Forecast accuracy for {owner}" if owner else "Forecast accuracy"
    win_rate = data.get("overall_win_rate", 0)

    return ToolResult(
        data=data,
        sources=[Source(
            type="forecast",
            id=f"accuracy_{owner or 'all'}",
            label=f"{label}: {win_rate}% win rate"
        )] if data.get("total_closed", 0) > 0 else []
    )


__all__ = [
    "tool_pipeline",
    "tool_pipeline_summary",
    "tool_pipeline_by_owner",
    "tool_upcoming_renewals",
    "tool_deals_at_risk",
    "tool_forecast",
    "tool_forecast_accuracy",
]
