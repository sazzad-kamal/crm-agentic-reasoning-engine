"""
Generic data formatting utilities.

Consolidates the repetitive format_*_section() functions from formatters.py
into a flexible formatter factory pattern.
"""

from typing import Optional, Callable, Any


class SectionFormatter:
    """
    Generic section formatter for CRM data.
    
    Reduces repetitive formatting code by providing a configurable
    template for formatting data sections.
    """
    
    def __init__(
        self,
        header: str,
        empty_message: str,
        item_formatter: Callable[[dict], str],
        max_items: int = 10,
        count_key: str = "count",
        items_key: str = "items",
        days_key: str = "days",
    ):
        """
        Initialize the formatter.
        
        Args:
            header: Section header text (without ===)
            empty_message: Message when no data available
            item_formatter: Function to format each item
            max_items: Maximum items to display
            count_key: Key for count in data dict
            items_key: Key for items list in data dict
            days_key: Key for days in data dict (for time-scoped sections)
        """
        self.header = header
        self.empty_message = empty_message
        self.item_formatter = item_formatter
        self.max_items = max_items
        self.count_key = count_key
        self.items_key = items_key
        self.days_key = days_key
    
    def format(self, data: Optional[dict]) -> str:
        """
        Format the data section.
        
        Args:
            data: Dict containing items and optional metadata
            
        Returns:
            Formatted section string
        """
        if not data:
            return ""
        
        items = data.get(self.items_key, [])
        if not items:
            return f"=== {self.header} ===\n{self.empty_message}"
        
        # Build header with optional count and days
        header_parts = [f"=== {self.header}"]
        
        count = data.get(self.count_key)
        days = data.get(self.days_key)
        
        if count is not None and days is not None:
            header_parts.append(f" ({count} found, last {days} days)")
        elif count is not None:
            header_parts.append(f" ({count})")
        elif days is not None:
            header_parts.append(f" (last {days} days)")
        
        header_parts.append(" ===")
        
        lines = ["".join(header_parts)]
        
        for item in items[:self.max_items]:
            lines.append(self.item_formatter(item))
        
        return "\n".join(lines)


def _format_date(date_str: str) -> str:
    """Extract date portion from datetime string."""
    if date_str and "T" in str(date_str):
        return str(date_str).split("T")[0]
    return str(date_str) if date_str else "N/A"


def _truncate(text: str, max_len: int = 100) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    text = str(text)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# =============================================================================
# Pre-configured formatters
# =============================================================================

def _format_activity(act: dict) -> str:
    """Format a single activity item."""
    due = _format_date(act.get("due_datetime", act.get("created_at")))
    return (
        f"- [{act.get('type', 'N/A')}] {act.get('subject', 'N/A')} "
        f"(Owner: {act.get('owner', 'N/A')}, Due: {due}, Status: {act.get('status', 'N/A')})"
    )


def _format_history(h: dict) -> str:
    """Format a single history item."""
    occurred = _format_date(h.get("occurred_at"))
    lines = [
        f"- [{h.get('type', 'N/A')}] {h.get('subject', 'N/A')} "
        f"(Date: {occurred}, Owner: {h.get('owner', 'N/A')})"
    ]
    if h.get("description"):
        lines.append(f"    Note: {_truncate(h.get('description', ''))}")
    return "\n".join(lines)


def _format_opportunity(opp: dict) -> str:
    """Format a single opportunity item."""
    close_date = opp.get("expected_close_date", "N/A")
    return (
        f"  - {opp.get('name', 'N/A')}: {opp.get('stage', 'N/A')} - "
        f"${opp.get('value', 0):,} (Close: {close_date})"
    )


def _format_renewal(r: dict) -> str:
    """Format a single renewal item."""
    return (
        f"- {r.get('name', 'N/A')} ({r.get('company_id', 'N/A')}): "
        f"Renewal {r.get('renewal_date', 'N/A')} | Plan: {r.get('plan', 'N/A')} | "
        f"Health: {r.get('health_flags', 'N/A')}"
    )


# Create singleton formatters for each section type
_ACTIVITIES_FORMATTER = SectionFormatter(
    header="RECENT ACTIVITIES",
    empty_message="No recent activities found.",
    item_formatter=_format_activity,
    max_items=8,
    items_key="activities",
)

_HISTORY_FORMATTER = SectionFormatter(
    header="HISTORY LOG",
    empty_message="No recent history entries.",
    item_formatter=_format_history,
    max_items=8,
    items_key="history",
)

_RENEWALS_FORMATTER = SectionFormatter(
    header="UPCOMING RENEWALS",
    empty_message="No renewals in the specified timeframe.",
    item_formatter=_format_renewal,
    max_items=10,
    items_key="renewals",
)


# =============================================================================
# Public API (backwards compatible)
# =============================================================================

def format_company_section(company_data: Optional[dict]) -> str:
    """Format company data for the prompt."""
    if not company_data:
        return ""
    
    company = company_data.get("company")
    if not company:
        return ""
    
    lines = [
        "=== COMPANY INFO ===",
        f"Name: {company.get('name', 'N/A')}",
        f"ID: {company.get('company_id', 'N/A')}",
        f"Status: {company.get('status', 'N/A')}",
        f"Plan: {company.get('plan', 'N/A')}",
        f"Industry: {company.get('industry', 'N/A')}",
        f"Region: {company.get('region', 'N/A')}",
        f"Account Owner: {company.get('account_owner', 'N/A')}",
        f"Renewal Date: {company.get('renewal_date', 'N/A')}",
        f"Health: {company.get('health_flags', 'N/A')}",
    ]
    
    contacts = company_data.get("contacts", [])
    if contacts:
        lines.append("\nKey Contacts:")
        for c in contacts[:3]:
            name = f"{c.get('first_name', '')} {c.get('last_name', '')}".strip()
            lines.append(f"  - {name} ({c.get('job_title', 'N/A')}): {c.get('email', 'N/A')}")
    
    return "\n".join(lines)


def format_activities_section(activities_data: Optional[dict]) -> str:
    """Format activities data for the prompt."""
    return _ACTIVITIES_FORMATTER.format(activities_data)


def format_history_section(history_data: Optional[dict]) -> str:
    """Format history data for the prompt."""
    return _HISTORY_FORMATTER.format(history_data)


def format_pipeline_section(pipeline_data: Optional[dict]) -> str:
    """Format pipeline data for the prompt."""
    if not pipeline_data:
        return ""
    
    summary = pipeline_data.get("summary", {})
    opps = pipeline_data.get("opportunities", [])
    
    if not summary.get("total_count"):
        return "=== PIPELINE ===\nNo open opportunities."
    
    lines = [
        f"=== PIPELINE SUMMARY ===",
        f"Total Open Deals: {summary.get('total_count', 0)}",
        f"Total Value: ${summary.get('total_value', 0):,.0f}",
        "\nBy Stage:"
    ]
    
    for stage, data in summary.get("stages", {}).items():
        lines.append(f"  - {stage}: {data.get('count', 0)} deals (${data.get('total_value', 0):,.0f})")
    
    if opps:
        lines.append("\nOpen Opportunities:")
        for opp in opps[:5]:
            lines.append(_format_opportunity(opp))
    
    return "\n".join(lines)


def format_renewals_section(renewals_data: Optional[dict]) -> str:
    """Format renewals data for the prompt."""
    return _RENEWALS_FORMATTER.format(renewals_data)


def format_docs_section(docs_answer: str) -> str:
    """Format docs RAG answer for the prompt."""
    if not docs_answer:
        return ""
    return f"=== DOCUMENTATION GUIDANCE ===\n{docs_answer}"
