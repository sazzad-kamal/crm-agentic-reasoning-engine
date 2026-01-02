"""
Company-related intent handlers.

Handles company_status, company_search, contacts, and attachments intents.
"""

from backend.agent.extractors import (
    extract_role_from_question,
    extract_company_criteria,
    extract_attachment_query,
)
from backend.agent.tools.company import (
    tool_search_contacts,
    tool_search_companies,
    tool_search_attachments,
)
from backend.agent.tools.pipeline import tool_pipeline
from backend.agent.tools.activity import tool_recent_activity, tool_recent_history

from backend.agent.handlers.common import (
    IntentContext,
    IntentResult,
    empty_raw_data,
    apply_tool_result,
    lookup_company,
    logger,
)


def handle_company_status(ctx: IntentContext) -> IntentResult:
    """Handle company_status and company-specific intents."""
    result = IntentResult(raw_data=empty_raw_data())

    query = ctx.resolved_company_id
    if not query and ctx.router_result:
        query = getattr(ctx.router_result, "company_name_query", None)

    logger.debug(f"[Data] Looking up company: {query}")
    if not lookup_company(result, query or ""):
        logger.info(f"[Data] Company not found: {query}")
        return result

    company_id = result.resolved_company_id
    logger.debug(f"[Data] Fetching data for {company_id}")

    # Fetch all company data
    apply_tool_result(
        result, tool_recent_activity(company_id, days=ctx.days),
        "activities_data", "activities"
    )
    apply_tool_result(
        result, tool_recent_history(company_id, days=ctx.days),
        "history_data", "history"
    )
    pipeline_result = tool_pipeline(company_id)
    apply_tool_result(result, pipeline_result, "pipeline_data", "opportunities")
    result.raw_data["pipeline_summary"] = pipeline_result.data.get("summary")

    logger.info(
        f"[Data] Fetched: activities={len(result.activities_data.get('activities', []))}, "
        f"history={len(result.history_data.get('history', []))}, "
        f"opps={len(result.pipeline_data.get('opportunities', []))}"
    )
    return result


def handle_company_search(ctx: IntentContext) -> IntentResult:
    """Handle company_search intent."""
    logger.debug("[Data] Searching companies")
    result = IntentResult(raw_data=empty_raw_data())

    segment, industry = extract_company_criteria(ctx.question)
    companies_result = tool_search_companies(segment=segment, industry=industry)
    apply_tool_result(result, companies_result, "company_data", "companies", limit=10)
    return result


def handle_contacts(ctx: IntentContext) -> IntentResult:
    """Handle contact_lookup and contact_search intents."""
    logger.debug("[Data] Handling contact query")
    result = IntentResult(raw_data=empty_raw_data(), resolved_company_id=ctx.resolved_company_id)

    role = extract_role_from_question(ctx.question)
    contacts_result = tool_search_contacts(
        company_id=ctx.resolved_company_id or "", role=role
    )
    apply_tool_result(result, contacts_result, "contacts_data", "contacts", limit=10)
    return result


def handle_attachments(ctx: IntentContext) -> IntentResult:
    """Handle attachments intent."""
    logger.debug("[Data] Searching attachments")
    result = IntentResult(raw_data=empty_raw_data(), resolved_company_id=ctx.resolved_company_id)

    query = extract_attachment_query(ctx.question)
    attachments_result = tool_search_attachments(query=query, company_id=ctx.resolved_company_id)
    apply_tool_result(result, attachments_result, "attachments_data", "attachments", limit=10)
    return result


__all__ = [
    "handle_company_status",
    "handle_company_search",
    "handle_contacts",
    "handle_attachments",
]
