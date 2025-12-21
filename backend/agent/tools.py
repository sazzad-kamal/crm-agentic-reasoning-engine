"""
Tool functions for the agentic layer.

Each tool returns both data and source citations.
Tools are pure Python and easy to test.
"""

from typing import Optional
from backend.agent.datastore import get_datastore, CRMDataStore
from backend.agent.schemas import Source, ToolResult


# =============================================================================
# Tool: Company Lookup
# =============================================================================

def tool_company_lookup(
    company_id_or_name: str,
    datastore: Optional[CRMDataStore] = None
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
    datastore: Optional[CRMDataStore] = None
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
    
    # Create sources for activities
    sources = []
    if activities:
        sources.append(
            Source(
                type="activities",
                id=company_id,
                label=f"Activities for {company_name} (last {days} days)"
            )
        )
    
    return ToolResult(
        data={
            "company_id": company_id,
            "company_name": company_name,
            "days": days,
            "count": len(activities),
            "activities": activities,
        },
        sources=sources
    )


# =============================================================================
# Tool: Recent History
# =============================================================================

def tool_recent_history(
    company_id: str,
    days: int = 90,
    limit: int = 20,
    datastore: Optional[CRMDataStore] = None
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
    
    sources = []
    if history:
        sources.append(
            Source(
                type="history",
                id=company_id,
                label=f"History for {company_name} (last {days} days)"
            )
        )
    
    return ToolResult(
        data={
            "company_id": company_id,
            "company_name": company_name,
            "days": days,
            "count": len(history),
            "history": history,
        },
        sources=sources
    )


# =============================================================================
# Tool: Pipeline
# =============================================================================

def tool_pipeline(
    company_id: str,
    datastore: Optional[CRMDataStore] = None
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
    
    sources = []
    if opportunities or summary.get("total_count", 0) > 0:
        sources.append(
            Source(
                type="opportunities",
                id=company_id,
                label=f"Pipeline for {company_name}"
            )
        )
    
    return ToolResult(
        data={
            "company_id": company_id,
            "company_name": company_name,
            "summary": summary,
            "opportunities": opportunities,
        },
        sources=sources
    )


# =============================================================================
# Tool: Upcoming Renewals
# =============================================================================

def tool_upcoming_renewals(
    days: int = 90,
    limit: int = 20,
    datastore: Optional[CRMDataStore] = None
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
    
    sources = []
    if renewals:
        sources.append(
            Source(
                type="renewals",
                id="upcoming",
                label=f"Upcoming renewals (next {days} days)"
            )
        )
    
    return ToolResult(
        data={
            "days": days,
            "count": len(renewals),
            "renewals": renewals,
        },
        sources=sources
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Tools")
    print("=" * 60)
    
    # Test company lookup
    print("\n1. Testing tool_company_lookup('Acme Manufacturing')...")
    result = tool_company_lookup("Acme Manufacturing")
    if result.data.get("found"):
        print(f"   Found: {result.data['company'].get('name')}")
        print(f"   Sources: {[s.label for s in result.sources]}")
    else:
        print(f"   Not found: {result.error}")
    
    # Test not found
    print("\n2. Testing tool_company_lookup('Unknown Corp')...")
    result = tool_company_lookup("Unknown Corp")
    print(f"   Found: {result.data.get('found')}")
    if result.data.get("close_matches"):
        print(f"   Close matches: {[c.get('name') for c in result.data['close_matches']]}")
    
    # Test recent activity
    print("\n3. Testing tool_recent_activity('ACME-MFG', days=365)...")
    result = tool_recent_activity("ACME-MFG", days=365)
    print(f"   Count: {result.data.get('count')}")
    for act in result.data.get("activities", [])[:2]:
        print(f"   - {act.get('type')}: {act.get('subject')}")
    
    # Test pipeline
    print("\n4. Testing tool_pipeline('ACME-MFG')...")
    result = tool_pipeline("ACME-MFG")
    summary = result.data.get("summary", {})
    print(f"   Total deals: {summary.get('total_count')}")
    print(f"   Total value: ${summary.get('total_value')}")
    
    # Test renewals
    print("\n5. Testing tool_upcoming_renewals(days=365)...")
    result = tool_upcoming_renewals(days=365)
    print(f"   Count: {result.data.get('count')}")
    for r in result.data.get("renewals", [])[:3]:
        print(f"   - {r.get('name')}: {r.get('renewal_date')}")
