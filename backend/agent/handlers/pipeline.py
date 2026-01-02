"""
Pipeline-related intent handlers.

Handles pipeline_summary, renewals, deals_at_risk, forecast, and forecast_accuracy intents.
"""

from backend.agent.tools.company import tool_accounts_needing_attention
from backend.agent.tools.pipeline import (
    tool_upcoming_renewals,
    tool_pipeline_summary,
    tool_pipeline_by_owner,
    tool_deals_at_risk,
    tool_forecast,
    tool_forecast_accuracy,
)

from backend.agent.handlers.common import (
    IntentContext,
    IntentResult,
    empty_raw_data,
    apply_tool_result,
    lookup_company,
    safe_extend,
    logger,
)


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
    "handle_pipeline_summary",
    "handle_renewals",
    "handle_deals_at_risk",
    "handle_forecast",
    "handle_forecast_accuracy",
]
