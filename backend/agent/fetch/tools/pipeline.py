"""
Pipeline-related intent handlers and tools.

Handles pipeline_summary, renewals, deals_at_risk, forecast, and forecast_accuracy intents.
Includes tool functions merged from tools/pipeline.py.
"""

from backend.agent.fetch.tools.common import (
    CRMDataStore,
    IntentContext,
    IntentResult,
    Source,
    ToolResult,
    apply_tool_result,
    empty_raw_data,
    logger,
    lookup_company,
    make_sources,
    safe_extend,
    with_datastore,
)

# =============================================================================
# Tool Functions (merged from tools/pipeline.py)
# =============================================================================


@with_datastore
def tool_pipeline(company_id: str, datastore: CRMDataStore | None = None) -> ToolResult:
    """Get pipeline summary and open opportunities for a company."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore

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
        sources=make_sources(
            has_pipeline, "opportunities", company_id, f"Pipeline for {company_name}"
        ),
    )


@with_datastore
def tool_pipeline_summary(datastore: CRMDataStore | None = None) -> ToolResult:
    """Get overall pipeline summary across all companies."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore
    summary = ds.get_all_pipeline_summary()

    return ToolResult(
        data=summary,
        sources=[
            Source(
                type="pipeline",
                id="all",
                label=f"Pipeline: {summary['total_count']} deals, ${summary['total_value']:,.0f}",
            )
        ]
        if summary.get("total_count", 0) > 0
        else [],
    )


@with_datastore
def tool_pipeline_by_owner(owner: str = "", datastore: CRMDataStore | None = None) -> ToolResult:
    """Get pipeline summary grouped by owner."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore
    data = ds.get_pipeline_by_owner(owner=owner or None)

    label = f"Pipeline for {owner}" if owner else "Pipeline by owner"

    return ToolResult(
        data=data,
        sources=[Source(type="pipeline", id=f"by_owner_{owner or 'all'}", label=label)]
        if data.get("total_count", 0) > 0
        else [],
    )


@with_datastore
def tool_upcoming_renewals(
    days: int = 90, limit: int = 20, owner: str = "", datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get companies with upcoming renewals."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore
    renewals = ds.get_upcoming_renewals(days=days, limit=limit, owner=owner or None)

    label = (
        f"Renewals for {owner} (next {days} days)"
        if owner
        else f"Upcoming renewals (next {days} days)"
    )

    return ToolResult(
        data={
            "days": days,
            "count": len(renewals),
            "owner_filter": owner or None,
            "renewals": renewals,
        },
        sources=make_sources(renewals, "renewals", "upcoming", label),
    )


@with_datastore
def tool_deals_at_risk(
    owner: str = "",
    days_threshold: int = 45,
    limit: int = 20,
    datastore: CRMDataStore | None = None,
) -> ToolResult:
    """Get deals that are at risk (stale, need attention)."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore
    deals = ds.get_deals_at_risk(owner=owner or None, days_threshold=days_threshold, limit=limit)

    label = (
        f"At-risk deals for {owner}"
        if owner
        else f"At-risk deals (>{days_threshold} days in stage)"
    )

    return ToolResult(
        data={
            "count": len(deals),
            "days_threshold": days_threshold,
            "owner_filter": owner or None,
            "deals": deals,
        },
        sources=make_sources(deals, "opportunities", "at_risk", label),
    )


@with_datastore
def tool_forecast(owner: str = "", datastore: CRMDataStore | None = None) -> ToolResult:
    """Get weighted pipeline forecast."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore
    data = ds.get_forecast(owner=owner or None)

    label = f"Forecast for {owner}" if owner else "Pipeline forecast"

    return ToolResult(
        data=data,
        sources=[
            Source(
                type="forecast",
                id=f"forecast_{owner or 'all'}",
                label=f"{label}: ${data.get('total_weighted', 0):,.0f} weighted",
            )
        ]
        if data.get("total_pipeline", 0) > 0
        else [],
    )


@with_datastore
def tool_forecast_accuracy(owner: str = "", datastore: CRMDataStore | None = None) -> ToolResult:
    """Get forecast accuracy metrics (win rate) based on historical closed deals."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore
    data = ds.get_forecast_accuracy(owner=owner or None)

    label = f"Forecast accuracy for {owner}" if owner else "Forecast accuracy"
    win_rate = data.get("overall_win_rate", 0)

    return ToolResult(
        data=data,
        sources=[
            Source(
                type="forecast",
                id=f"accuracy_{owner or 'all'}",
                label=f"{label}: {win_rate}% win rate",
            )
        ]
        if data.get("total_closed", 0) > 0
        else [],
    )


# =============================================================================
# Intent Handlers
# =============================================================================


def handle_pipeline_summary(ctx: IntentContext) -> IntentResult:
    """Handle pipeline_summary intent."""
    logger.debug(f"[Data] Fetching pipeline summary (owner={ctx.owner})")
    result = IntentResult(raw_data=empty_raw_data())

    # Use owner-filtered pipeline tool when owner is set
    if ctx.owner:
        pipeline_result = tool_pipeline_by_owner(owner=ctx.owner)
        result.pipeline_data = pipeline_result.data
        safe_extend(result.sources, pipeline_result.sources)
        result.raw_data["pipeline_summary"] = pipeline_result.data.get("summary", {})
        result.raw_data["opportunities"] = pipeline_result.data.get("opportunities", [])[:8]
    else:
        # Manager view - all opportunities
        summary_result = tool_pipeline_summary()
        result.pipeline_data = summary_result.data
        safe_extend(result.sources, summary_result.sources)
        result.raw_data["pipeline_summary"] = {
            "total_count": result.pipeline_data.get("total_count"),
            "total_value": result.pipeline_data.get("total_value"),
            "by_stage": result.pipeline_data.get("by_stage", []),
        }
        result.raw_data["opportunities"] = result.pipeline_data.get("top_opportunities", [])[:8]
    return result


def handle_renewals(ctx: IntentContext) -> IntentResult:
    """Handle renewals intent."""
    logger.debug(f"[Data] Fetching renewals for next {ctx.days} days (owner={ctx.owner})")
    result = IntentResult(raw_data=empty_raw_data(), resolved_company_id=ctx.resolved_company_id)

    if ctx.resolved_company_id:
        lookup_company(result, ctx.resolved_company_id)

    renewals_result = tool_upcoming_renewals(days=ctx.days, owner=ctx.owner or "")
    apply_tool_result(result, renewals_result, "renewals_data", "renewals")
    return result


def handle_deals_at_risk(ctx: IntentContext) -> IntentResult:
    """Handle deals_at_risk intent - shows stalled/at-risk deals and accounts needing attention."""
    # Import here to avoid circular imports
    from backend.agent.fetch.tools.company import tool_accounts_needing_attention

    logger.debug(f"[Data] Fetching at-risk deals and accounts (owner={ctx.owner})")
    result = IntentResult(raw_data=empty_raw_data())

    # Get deals at risk
    risk_result = tool_deals_at_risk(owner=ctx.owner or "")
    result.pipeline_data = risk_result.data
    safe_extend(result.sources, risk_result.sources)
    result.raw_data["opportunities"] = risk_result.data.get("deals", [])[:8]

    # Get renewals for context
    renewals_result = tool_upcoming_renewals(days=ctx.days, owner=ctx.owner or "")
    apply_tool_result(result, renewals_result, "renewals_data", "renewals")

    # Get accounts needing attention
    accounts_result = tool_accounts_needing_attention(owner=ctx.owner or "")
    safe_extend(result.sources, accounts_result.sources)
    result.raw_data["companies"] = accounts_result.data.get("accounts", [])[:8]

    result.raw_data["pipeline_summary"] = {
        "at_risk_count": result.pipeline_data.get("count", 0),
        "at_risk_value": result.pipeline_data.get("total_value", 0),
        "accounts_needing_attention": accounts_result.data.get("count", 0),
    }
    return result


def handle_forecast(ctx: IntentContext) -> IntentResult:
    """Handle forecast intent - shows weighted pipeline projections."""
    logger.debug(f"[Data] Fetching pipeline forecast (owner={ctx.owner})")
    result = IntentResult(raw_data=empty_raw_data())

    forecast_result = tool_forecast(owner=ctx.owner or "")
    result.pipeline_data = forecast_result.data
    safe_extend(result.sources, forecast_result.sources)
    result.raw_data["pipeline_summary"] = forecast_result.data
    result.raw_data["opportunities"] = forecast_result.data.get("top_opportunities", [])[:8]
    return result


def handle_forecast_accuracy(ctx: IntentContext) -> IntentResult:
    """Handle forecast_accuracy intent - shows win rate metrics."""
    logger.debug(f"[Data] Fetching forecast accuracy (owner={ctx.owner})")
    result = IntentResult(raw_data=empty_raw_data())

    accuracy_result = tool_forecast_accuracy(owner=ctx.owner or "")
    result.pipeline_data = accuracy_result.data
    safe_extend(result.sources, accuracy_result.sources)
    result.raw_data["pipeline_summary"] = accuracy_result.data
    result.raw_data["analytics"] = accuracy_result.data
    return result


__all__ = [
    # Handlers
    "handle_pipeline_summary",
    "handle_renewals",
    "handle_deals_at_risk",
    "handle_forecast",
    "handle_forecast_accuracy",
    # Tools (for backward compatibility)
    "tool_pipeline",
    "tool_pipeline_summary",
    "tool_pipeline_by_owner",
    "tool_upcoming_renewals",
    "tool_deals_at_risk",
    "tool_forecast",
    "tool_forecast_accuracy",
]
