"""
Company-related intent handlers and tools.

Handles company_status, company_search, contacts, and attachments intents.
Includes tool functions merged from tools/company.py.
"""

from backend.agent.fetch.tools.common import (
    CRMDataStore,
    IntentContext,
    IntentResult,
    ToolResult,
    apply_tool_result,
    empty_raw_data,
    logger,
    lookup_company,
    make_sources,
    with_datastore,
)
from backend.agent.fetch.tools.extractors import (
    extract_attachment_query,
    extract_company_criteria,
    extract_role_from_question,
)

# =============================================================================
# Tool Functions (merged from tools/company.py)
# =============================================================================


@with_datastore
def tool_search_companies(
    query: str = "",
    industry: str = "",
    segment: str = "",
    status: str = "",
    region: str = "",
    limit: int = 20,
    datastore: CRMDataStore | None = None,
) -> ToolResult:
    """Search companies by various criteria."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore

    companies = ds.search_companies(
        query=query, industry=industry, segment=segment, status=status, region=region, limit=limit
    )

    search_desc = []
    if query:
        search_desc.append(f"name='{query}'")
    if industry:
        search_desc.append(f"industry='{industry}'")
    if segment:
        search_desc.append(f"segment='{segment}'")
    if status:
        search_desc.append(f"status='{status}'")
    if region:
        search_desc.append(f"region='{region}'")

    label = f"Companies: {', '.join(search_desc)}" if search_desc else "All companies"

    return ToolResult(
        data={
            "count": len(companies),
            "companies": companies,
            "filters": {
                "query": query,
                "industry": industry,
                "segment": segment,
                "status": status,
                "region": region,
            },
        },
        sources=make_sources(companies, "companies", "search", label),
    )


@with_datastore
def tool_search_contacts(
    query: str = "",
    role: str = "",
    job_title: str = "",
    company_id: str = "",
    limit: int = 20,
    datastore: CRMDataStore | None = None,
) -> ToolResult:
    """Search contacts by name, role, job title, or company."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore

    contacts = ds.search_contacts(
        query=query, role=role, job_title=job_title, company_id=company_id, limit=limit
    )

    search_desc = []
    if query:
        search_desc.append(f"name/email='{query}'")
    if role:
        search_desc.append(f"role='{role}'")
    if job_title:
        search_desc.append(f"title='{job_title}'")
    if company_id:
        search_desc.append(f"company='{company_id}'")

    label = f"Contacts: {', '.join(search_desc)}" if search_desc else "All contacts"

    return ToolResult(
        data={
            "count": len(contacts),
            "contacts": contacts,
            "filters": {
                "query": query,
                "role": role,
                "job_title": job_title,
                "company_id": company_id,
            },
        },
        sources=make_sources(contacts, "contacts", "search", label),
    )


@with_datastore
def tool_search_attachments(
    query: str = "",
    company_id: str = "",
    file_type: str = "",
    limit: int = 20,
    datastore: CRMDataStore | None = None,
) -> ToolResult:
    """Search attachments/documents by title, company, or file type."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore

    attachments = ds.search_attachments(
        query=query, company_id=company_id, file_type=file_type, limit=limit
    )

    search_desc = []
    if query:
        search_desc.append(f"title='{query}'")
    if company_id:
        search_desc.append(f"company='{company_id}'")
    if file_type:
        search_desc.append(f"type='{file_type}'")

    label = f"Attachments: {', '.join(search_desc)}" if search_desc else "All attachments"

    return ToolResult(
        data={
            "count": len(attachments),
            "attachments": attachments,
            "filters": {"query": query, "company_id": company_id, "file_type": file_type},
        },
        sources=make_sources(attachments, "attachments", "search", label),
    )


@with_datastore
def tool_accounts_needing_attention(
    owner: str = "", limit: int = 20, datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get accounts that need immediate attention (trial, churned, at-risk)."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore

    accounts = ds.get_accounts_needing_attention(owner=owner or None, limit=limit)

    label = (
        f"Accounts needing attention for {owner}"
        if owner
        else "Accounts needing immediate attention"
    )

    return ToolResult(
        data={
            "count": len(accounts),
            "owner_filter": owner or None,
            "accounts": accounts,
        },
        sources=make_sources(accounts, "companies", "attention", label),
    )


# =============================================================================
# Intent Handlers
# =============================================================================


def handle_company_status(ctx: IntentContext) -> IntentResult:
    """Handle company_status and company-specific intents."""
    # Import here to avoid circular imports
    from backend.agent.fetch.tools.activity import tool_recent_activity, tool_recent_history
    from backend.agent.fetch.tools.pipeline import tool_pipeline

    result = IntentResult(raw_data=empty_raw_data())

    company_id = ctx.resolved_company_id

    logger.debug(f"[Data] Looking up company: {company_id}")
    if not lookup_company(result, company_id or ""):
        logger.info(f"[Data] Company not found: {company_id}")
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
        f"[Data] Fetched: activities={len((result.activities_data or {}).get('activities', []))}, "
        f"history={len((result.history_data or {}).get('history', []))}, "
        f"opps={len((result.pipeline_data or {}).get('opportunities', []))}"
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
    # Handlers
    "handle_company_status",
    "handle_company_search",
    "handle_contacts",
    "handle_attachments",
    # Tools
    "tool_search_companies",
    "tool_search_contacts",
    "tool_search_attachments",
    "tool_accounts_needing_attention",
]
