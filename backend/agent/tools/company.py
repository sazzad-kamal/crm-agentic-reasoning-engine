"""
Company and contact tools.
"""

from backend.agent.datastore import CRMDataStore, get_datastore
from backend.agent.schemas import Source, ToolResult
from backend.agent.tools.base import make_sources, with_datastore


@with_datastore
def tool_company_lookup(
    company_id_or_name: str,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Look up company information by ID or name."""
    ds = datastore  # Already resolved by decorator

    company_id = ds.resolve_company_id(company_id_or_name)

    if not company_id:
        matches = ds.get_company_name_matches(company_id_or_name, limit=5)
        return ToolResult(
            data={"found": False, "query": company_id_or_name, "close_matches": matches},
            sources=[],
            error=f"Company '{company_id_or_name}' not found"
        )

    company = ds.get_company(company_id)
    if not company:
        return ToolResult(
            data={"found": False, "query": company_id_or_name},
            sources=[],
            error=f"Company '{company_id}' not found in database"
        )

    contacts = ds.get_contacts_for_company(company_id, limit=5)

    return ToolResult(
        data={"found": True, "company": company, "contacts": contacts},
        sources=[Source(type="company", id=company_id, label=company.get("name", company_id))]
    )


@with_datastore
def tool_search_companies(
    query: str = "",
    industry: str = "",
    segment: str = "",
    status: str = "",
    region: str = "",
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Search companies by various criteria."""
    ds = datastore

    companies = ds.search_companies(
        query=query, industry=industry, segment=segment,
        status=status, region=region, limit=limit
    )

    search_desc = []
    if query: search_desc.append(f"name='{query}'")
    if industry: search_desc.append(f"industry='{industry}'")
    if segment: search_desc.append(f"segment='{segment}'")
    if status: search_desc.append(f"status='{status}'")
    if region: search_desc.append(f"region='{region}'")

    label = f"Companies: {', '.join(search_desc)}" if search_desc else "All companies"

    return ToolResult(
        data={
            "count": len(companies),
            "companies": companies,
            "filters": {"query": query, "industry": industry, "segment": segment, "status": status, "region": region},
        },
        sources=make_sources(companies, "companies", "search", label)
    )


@with_datastore
def tool_contact_lookup(
    contact_id: str,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Look up a specific contact by ID."""
    ds = datastore

    contact = ds.get_contact(contact_id)
    if not contact:
        return ToolResult(
            data={"found": False, "contact_id": contact_id},
            sources=[],
            error=f"Contact '{contact_id}' not found"
        )

    company = ds.get_company(contact.get("company_id", ""))

    return ToolResult(
        data={"found": True, "contact": contact, "company": company},
        sources=[Source(
            type="contact", id=contact_id,
            label=f"{contact.get('first_name', '')} {contact.get('last_name', '')}"
        )]
    )


@with_datastore
def tool_search_contacts(
    query: str = "",
    role: str = "",
    job_title: str = "",
    company_id: str = "",
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Search contacts by name, role, job title, or company."""
    ds = datastore

    contacts = ds.search_contacts(
        query=query, role=role, job_title=job_title,
        company_id=company_id, limit=limit
    )

    search_desc = []
    if query: search_desc.append(f"name/email='{query}'")
    if role: search_desc.append(f"role='{role}'")
    if job_title: search_desc.append(f"title='{job_title}'")
    if company_id: search_desc.append(f"company='{company_id}'")

    label = f"Contacts: {', '.join(search_desc)}" if search_desc else "All contacts"

    return ToolResult(
        data={
            "count": len(contacts),
            "contacts": contacts,
            "filters": {"query": query, "role": role, "job_title": job_title, "company_id": company_id},
        },
        sources=make_sources(contacts, "contacts", "search", label)
    )


@with_datastore
def tool_group_members(
    group_id: str,
    limit: int = 50,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Get companies that belong to a specific group/segment."""
    ds = datastore

    group = ds.get_group(group_id)
    if not group:
        all_groups = ds.get_all_groups()
        return ToolResult(
            data={
                "found": False, "group_id": group_id,
                "available_groups": [{"group_id": g["group_id"], "name": g["name"]} for g in all_groups],
            },
            sources=[],
            error=f"Group '{group_id}' not found"
        )

    members = ds.get_group_members(group_id, limit=limit)

    return ToolResult(
        data={"found": True, "group": group, "count": len(members), "members": members},
        sources=[Source(type="group", id=group_id, label=group.get("name", group_id))] if members else []
    )


@with_datastore
def tool_list_groups(datastore: CRMDataStore | None = None) -> ToolResult:
    """List all available groups/segments."""
    ds = datastore
    groups = ds.get_all_groups()

    return ToolResult(
        data={"count": len(groups), "groups": groups},
        sources=make_sources(groups, "groups", "all", "All groups/segments")
    )


@with_datastore
def tool_search_attachments(
    query: str = "",
    company_id: str = "",
    file_type: str = "",
    limit: int = 20,
    datastore: CRMDataStore | None = None
) -> ToolResult:
    """Search attachments/documents by title, company, or file type."""
    ds = datastore

    attachments = ds.search_attachments(query=query, company_id=company_id, file_type=file_type, limit=limit)

    search_desc = []
    if query: search_desc.append(f"title='{query}'")
    if company_id: search_desc.append(f"company='{company_id}'")
    if file_type: search_desc.append(f"type='{file_type}'")

    label = f"Attachments: {', '.join(search_desc)}" if search_desc else "All attachments"

    return ToolResult(
        data={
            "count": len(attachments),
            "attachments": attachments,
            "filters": {"query": query, "company_id": company_id, "file_type": file_type},
        },
        sources=make_sources(attachments, "attachments", "search", label)
    )


__all__ = [
    "tool_company_lookup",
    "tool_search_companies",
    "tool_contact_lookup",
    "tool_search_contacts",
    "tool_group_members",
    "tool_list_groups",
    "tool_search_attachments",
]
