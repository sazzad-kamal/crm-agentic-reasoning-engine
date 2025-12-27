"""
Tool functions for the agentic layer.

Each tool returns both data and source citations.
Tools are pure Python and easy to test.
"""

from backend.agent.datastore import get_datastore, CRMDataStore
from backend.agent.schemas import Source, ToolResult


def _make_sources(
    data: list | dict | None,
    source_type: str,
    source_id: str,
    label: str,
) -> list[Source]:
    """Create a source list if data is non-empty. Reduces repetitive conditional blocks."""
    if data:  # Works for non-empty list, dict with items, or truthy value
        return [Source(type=source_type, id=source_id, label=label)]
    return []


# =============================================================================
# Tool: Company Lookup
# =============================================================================

def tool_company_lookup(
    company_id_or_name: str,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Look up company information by ID or name.
    
    Args:
        company_id_or_name: Company ID or name to look up
        datastore: Optional datastore instance (for testing)
        
    Returns:
        ToolResult with company data and sources
    """
    ds = datastore or get_datastore()
    
    # Resolve to company ID
    company_id = ds.resolve_company_id(company_id_or_name)
    
    if not company_id:
        # Try to find close matches
        matches = ds.get_company_name_matches(company_id_or_name, limit=5)
        
        return ToolResult(
            data={
                "found": False,
                "query": company_id_or_name,
                "close_matches": matches,
            },
            sources=[],
            error=f"Company '{company_id_or_name}' not found"
        )
    
    # Get company details
    company = ds.get_company(company_id)
    
    if not company:
        return ToolResult(
            data={"found": False, "query": company_id_or_name},
            sources=[],
            error=f"Company '{company_id}' not found in database"
        )
    
    # Get contacts for the company
    contacts = ds.get_contacts_for_company(company_id, limit=5)
    
    return ToolResult(
        data={
            "found": True,
            "company": company,
            "contacts": contacts,
        },
        sources=[
            Source(
                type="company",
                id=company_id,
                label=company.get("name", company_id)
            )
        ]
    )


# =============================================================================
# Tool: Recent Activity
# =============================================================================

def tool_recent_activity(
    company_id: str,
    days: int = 90,
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Get recent activities for a company.
    
    Args:
        company_id: The company ID (must be resolved first)
        days: Number of days to look back
        limit: Maximum activities to return
        datastore: Optional datastore instance
        
    Returns:
        ToolResult with activities and sources
    """
    ds = datastore or get_datastore()
    
    activities = ds.get_recent_activities(company_id, days=days, limit=limit)
    
    # Get company name for labeling
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
        sources=_make_sources(
            activities, "activities", company_id,
            f"Activities for {company_name} (last {days} days)"
        )
    )


# =============================================================================
# Tool: Recent History
# =============================================================================

def tool_recent_history(
    company_id: str,
    days: int = 90,
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Get recent history entries (calls, emails, notes) for a company.
    
    Args:
        company_id: The company ID
        days: Number of days to look back
        limit: Maximum entries to return
        datastore: Optional datastore instance
        
    Returns:
        ToolResult with history and sources
    """
    ds = datastore or get_datastore()
    
    history = ds.get_recent_history(company_id, days=days, limit=limit)
    
    # Get company name
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
        sources=_make_sources(
            history, "history", company_id,
            f"History for {company_name} (last {days} days)"
        )
    )


# =============================================================================
# Tool: Pipeline
# =============================================================================

def tool_pipeline(
    company_id: str,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Get pipeline summary and open opportunities for a company.
    
    Args:
        company_id: The company ID
        datastore: Optional datastore instance
        
    Returns:
        ToolResult with pipeline data and sources
    """
    ds = datastore or get_datastore()
    
    # Get summary
    summary = ds.get_pipeline_summary(company_id)
    
    # Get individual opportunities
    opportunities = ds.get_open_opportunities(company_id, limit=20)
    
    # Get company name
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
        sources=_make_sources(
            has_pipeline, "opportunities", company_id,
            f"Pipeline for {company_name}"
        )
    )


# =============================================================================
# Tool: Upcoming Renewals
# =============================================================================

def tool_upcoming_renewals(
    days: int = 90,
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Get companies with upcoming renewals.
    
    Args:
        days: Number of days to look ahead
        limit: Maximum results to return
        datastore: Optional datastore instance
        
    Returns:
        ToolResult with renewals and sources
    """
    ds = datastore or get_datastore()
    
    renewals = ds.get_upcoming_renewals(days=days, limit=limit)
    
    return ToolResult(
        data={
            "days": days,
            "count": len(renewals),
            "renewals": renewals,
        },
        sources=_make_sources(
            renewals, "renewals", "upcoming",
            f"Upcoming renewals (next {days} days)"
        )
    )


# =============================================================================
# Tool: Contact Lookup
# =============================================================================

def tool_contact_lookup(
    contact_id: str,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Look up a specific contact by ID.
    
    Args:
        contact_id: The contact ID
        datastore: Optional datastore instance
        
    Returns:
        ToolResult with contact data
    """
    ds = datastore or get_datastore()
    
    contact = ds.get_contact(contact_id)
    
    if not contact:
        return ToolResult(
            data={"found": False, "contact_id": contact_id},
            sources=[],
            error=f"Contact '{contact_id}' not found"
        )
    
    # Get company info
    company = ds.get_company(contact.get("company_id", ""))
    
    return ToolResult(
        data={
            "found": True,
            "contact": contact,
            "company": company,
        },
        sources=[Source(
            type="contact",
            id=contact_id,
            label=f"{contact.get('first_name', '')} {contact.get('last_name', '')}"
        )]
    )


# =============================================================================
# Tool: Search Contacts
# =============================================================================

def tool_search_contacts(
    query: str = "",
    role: str = "",
    job_title: str = "",
    company_id: str = "",
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Search contacts by name, role, job title, or company.
    
    Args:
        query: Search term for name/email
        role: Filter by role (e.g., "Decision Maker", "Technical Contact")
        job_title: Filter by job title (e.g., "VP", "Manager")
        company_id: Filter by company
        limit: Max results
        datastore: Optional datastore instance
        
    Returns:
        ToolResult with matching contacts
    """
    ds = datastore or get_datastore()
    
    contacts = ds.search_contacts(
        query=query,
        role=role,
        job_title=job_title,
        company_id=company_id,
        limit=limit
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
            "filters": {"query": query, "role": role, "job_title": job_title, "company_id": company_id},
        },
        sources=_make_sources(contacts, "contacts", "search", label)
    )


# =============================================================================
# Tool: Search Companies
# =============================================================================

def tool_search_companies(
    query: str = "",
    industry: str = "",
    segment: str = "",
    status: str = "",
    region: str = "",
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Search companies by various criteria.
    
    Args:
        query: Search term for company name
        industry: Filter by industry (e.g., "Healthcare", "Software")
        segment: Filter by segment (e.g., "SMB", "Mid-market", "Enterprise")
        status: Filter by status (e.g., "Active", "Churned")
        region: Filter by region
        limit: Max results
        datastore: Optional datastore instance
        
    Returns:
        ToolResult with matching companies
    """
    ds = datastore or get_datastore()
    
    companies = ds.search_companies(
        query=query,
        industry=industry,
        segment=segment,
        status=status,
        region=region,
        limit=limit
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
            "filters": {"query": query, "industry": industry, "segment": segment, "status": status, "region": region},
        },
        sources=_make_sources(companies, "companies", "search", label)
    )


# =============================================================================
# Tool: Group Members
# =============================================================================

def tool_group_members(
    group_id: str,
    limit: int = 50,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Get companies that belong to a specific group/segment.
    
    Args:
        group_id: The group ID
        limit: Max results
        datastore: Optional datastore instance
        
    Returns:
        ToolResult with group info and member companies
    """
    ds = datastore or get_datastore()
    
    group = ds.get_group(group_id)
    
    if not group:
        # Try to find matching groups
        all_groups = ds.get_all_groups()
        return ToolResult(
            data={
                "found": False,
                "group_id": group_id,
                "available_groups": [{"group_id": g["group_id"], "name": g["name"]} for g in all_groups],
            },
            sources=[],
            error=f"Group '{group_id}' not found"
        )
    
    members = ds.get_group_members(group_id, limit=limit)
    
    return ToolResult(
        data={
            "found": True,
            "group": group,
            "count": len(members),
            "members": members,
        },
        sources=[Source(
            type="group",
            id=group_id,
            label=group.get("name", group_id)
        )] if members else []
    )


# =============================================================================
# Tool: List Groups
# =============================================================================

def tool_list_groups(
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    List all available groups/segments.
    
    Returns:
        ToolResult with all groups
    """
    ds = datastore or get_datastore()
    
    groups = ds.get_all_groups()
    
    return ToolResult(
        data={
            "count": len(groups),
            "groups": groups,
        },
        sources=_make_sources(groups, "groups", "all", "All groups/segments")
    )


# =============================================================================
# Tool: Search Attachments
# =============================================================================

def tool_search_attachments(
    query: str = "",
    company_id: str = "",
    file_type: str = "",
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Search attachments/documents by title, company, or file type.
    
    Args:
        query: Search term for title/summary
        company_id: Filter by company
        file_type: Filter by file type (e.g., "pdf", "xlsx")
        limit: Max results
        datastore: Optional datastore instance
        
    Returns:
        ToolResult with matching attachments
    """
    ds = datastore or get_datastore()
    
    attachments = ds.search_attachments(
        query=query,
        company_id=company_id,
        file_type=file_type,
        limit=limit
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
        sources=_make_sources(attachments, "attachments", "search", label)
    )


# =============================================================================
# Tool: Pipeline Summary (All Companies)
# =============================================================================

def tool_pipeline_summary(
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Get overall pipeline summary across all companies.
    
    Returns:
        ToolResult with aggregate pipeline stats
    """
    ds = datastore or get_datastore()
    
    summary = ds.get_all_pipeline_summary()
    
    return ToolResult(
        data=summary,
        sources=[Source(
            type="pipeline",
            id="all",
            label=f"Pipeline: {summary['total_count']} deals, ${summary['total_value']:,.0f}"
        )] if summary.get("total_count", 0) > 0 else []
    )


# =============================================================================
# Tool: Search Activities
# =============================================================================

def tool_search_activities(
    activity_type: str = "",
    days: int = 30,
    company_id: str = "",
    limit: int = 30,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """
    Search activities by type, date range, or company.
    
    Args:
        activity_type: Filter by type (e.g., "Demo", "Meeting", "Call")
        days: Look back N days (default 30)
        company_id: Filter by company
        limit: Max results
        datastore: Optional datastore instance
        
    Returns:
        ToolResult with matching activities
    """
    ds = datastore or get_datastore()
    
    activities = ds.search_activities(
        activity_type=activity_type,
        days=days,
        company_id=company_id,
        limit=limit
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
        sources=_make_sources(activities, "activities", "search", label)
    )


__all__ = [
    "tool_company_lookup",
    "tool_recent_activity",
    "tool_recent_history",
    "tool_pipeline",
    "tool_upcoming_renewals",
    "tool_contact_lookup",
    "tool_search_contacts",
    "tool_search_companies",
    "tool_group_members",
    "tool_list_groups",
    "tool_search_attachments",
    "tool_pipeline_summary",
    "tool_search_activities",
]


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Tools")
    print("=" * 60)
    
    # Test company lookup
    print("\n1. tool_company_lookup('Acme Manufacturing')...")
    result = tool_company_lookup("Acme Manufacturing")
    if result.data.get("found"):
        print(f"   Found: {result.data['company'].get('name')}")
    else:
        print(f"   Not found: {result.error}")
    
    # Test recent activity
    print("\n2. tool_recent_activity('ACME-MFG', days=365)...")
    result = tool_recent_activity("ACME-MFG", days=365)
    print(f"   Count: {result.data.get('count')}")
    
    # Test pipeline
    print("\n3. tool_pipeline('ACME-MFG')...")
    result = tool_pipeline("ACME-MFG")
    print(f"   Total deals: {result.data.get('summary', {}).get('total_count')}")
    
    # Test renewals
    print("\n4. tool_upcoming_renewals(days=365)...")
    result = tool_upcoming_renewals(days=365)
    print(f"   Count: {result.data.get('count')}")
    
    # Test search contacts
    print("\n5. tool_search_contacts(role='Decision Maker')...")
    result = tool_search_contacts(role="Decision Maker")
    print(f"   Count: {result.data.get('count')}")
    for c in result.data.get("contacts", [])[:3]:
        print(f"   - {c.get('first_name')} {c.get('last_name')} ({c.get('job_title')})")
    
    # Test search companies
    print("\n6. tool_search_companies(industry='Software')...")
    result = tool_search_companies(industry="Software")
    print(f"   Count: {result.data.get('count')}")
    for co in result.data.get("companies", [])[:3]:
        print(f"   - {co.get('name')} ({co.get('segment')})")
    
    # Test list groups
    print("\n7. tool_list_groups()...")
    result = tool_list_groups()
    print(f"   Count: {result.data.get('count')}")
    for g in result.data.get("groups", [])[:3]:
        print(f"   - {g.get('group_id')}: {g.get('name')}")
    
    # Test pipeline summary
    print("\n8. tool_pipeline_summary()...")
    result = tool_pipeline_summary()
    print(f"   Total: {result.data.get('total_count')} deals, ${result.data.get('total_value', 0):,.0f}")
    
    # Test search attachments
    print("\n9. tool_search_attachments(query='proposal')...")
    result = tool_search_attachments(query="proposal")
    print(f"   Count: {result.data.get('count')}")
    for a in result.data.get("attachments", [])[:3]:
        print(f"   - {a.get('title')} ({a.get('file_type')})")
