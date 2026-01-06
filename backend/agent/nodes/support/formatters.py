"""
Generic data formatting utilities.

Uses a declarative formatter factory pattern for consistent, maintainable
formatting of CRM data sections.
"""

from typing import Callable, Any
from datetime import datetime, timedelta


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
        item_formatter: Callable[[dict[str, Any]], str],
        max_items: int = 10,
        count_key: str = "count",
        items_key: str = "items",
        days_key: str = "days",
    ):
        self.header = header
        self.empty_message = empty_message
        self.item_formatter = item_formatter
        self.max_items = max_items
        self.count_key = count_key
        self.items_key = items_key
        self.days_key = days_key

    def format(self, data: dict[str, Any] | None) -> str:
        """Format the data section."""
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
        for item in items[: self.max_items]:
            lines.append(self.item_formatter(item))

        return "\n".join(lines)


# =============================================================================
# Helper functions
# =============================================================================


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
    return f"{text[:max_len]}..." if len(text) > max_len else text


# =============================================================================
# Item formatters (used by SectionFormatter)
# =============================================================================


def _format_activity(act: dict) -> str:
    due = _format_date(act.get("due_datetime") or act.get("created_at") or "")
    return (
        f"- [{act.get('type', 'N/A')}] {act.get('subject', 'N/A')} "
        f"(Owner: {act.get('owner', 'N/A')}, Due: {due}, Status: {act.get('status', 'N/A')})"
    )


def _format_history(h: dict) -> str:
    occurred = _format_date(h.get("occurred_at") or "")
    lines = [
        f"- [{h.get('type', 'N/A')}] {h.get('subject', 'N/A')} "
        f"(Date: {occurred}, Owner: {h.get('owner', 'N/A')})"
    ]
    if h.get("description"):
        lines.append(f"    Note: {_truncate(h.get('description', ''))}")
    return "\n".join(lines)


def _format_opportunity(opp: dict) -> str:
    close_date = opp.get("expected_close_date", "N/A")
    return (
        f"  - {opp.get('name', 'N/A')}: {opp.get('stage', 'N/A')} - "
        f"${opp.get('value', 0):,} (Close: {close_date})"
    )


def _format_renewal(r: dict) -> str:
    return (
        f"- {r.get('name', 'N/A')} ({r.get('company_id', 'N/A')}): "
        f"Renewal {r.get('renewal_date', 'N/A')} | Plan: {r.get('plan', 'N/A')} | "
        f"Health: {r.get('health_flags', 'N/A')}"
    )


def _format_contact(c: dict) -> str:
    name = f"{c.get('first_name', '')} {c.get('last_name', '')}".strip()
    return (
        f"- {name}: {c.get('job_title', 'N/A')} at {c.get('company_id', 'N/A')} | "
        f"Role: {c.get('contact_role', 'N/A')} | Email: {c.get('email', 'N/A')}"
    )


def _format_attachment(a: dict) -> str:
    return (
        f"- {a.get('name', 'N/A')} ({a.get('file_type', 'N/A')}): "
        f"{_truncate(a.get('description', ''), 80)} | Company: {a.get('company_id', 'N/A')}"
    )


def _format_company_search_item(c: dict) -> str:
    return (
        f"- {c.get('name', 'N/A')} ({c.get('company_id', 'N/A')}): "
        f"{c.get('industry', 'N/A')} | {c.get('segment', 'N/A')} | "
        f"{c.get('status', 'N/A')} | Health: {c.get('health_flags', 'N/A')}"
    )


# =============================================================================
# Declarative Formatters Registry
# =============================================================================


FORMATTERS = {
    "activities": SectionFormatter(
        header="RECENT ACTIVITIES",
        empty_message="No recent activities found.",
        item_formatter=_format_activity,
        max_items=8,
        items_key="activities",
    ),
    "history": SectionFormatter(
        header="HISTORY LOG",
        empty_message="No recent history entries.",
        item_formatter=_format_history,
        max_items=8,
        items_key="history",
    ),
    "renewals": SectionFormatter(
        header="UPCOMING RENEWALS",
        empty_message="No renewals in the specified timeframe.",
        item_formatter=_format_renewal,
        max_items=10,
        items_key="renewals",
    ),
    "contacts": SectionFormatter(
        header="CONTACTS",
        empty_message="No contacts found matching the criteria.",
        item_formatter=_format_contact,
        max_items=10,
        items_key="contacts",
    ),
    "attachments": SectionFormatter(
        header="ATTACHMENTS",
        empty_message="No attachments found matching the criteria.",
        item_formatter=_format_attachment,
        max_items=10,
        items_key="attachments",
    ),
    "companies": SectionFormatter(
        header="COMPANY SEARCH RESULTS",
        empty_message="No companies found matching the criteria.",
        item_formatter=_format_company_search_item,
        max_items=10,
        items_key="companies",
    ),
}


def format_section(section_type: str, data: dict | None) -> str:
    """Generic section formatter using the registry."""
    if section_type not in FORMATTERS:
        raise ValueError(f"Unknown section type: {section_type}")
    return FORMATTERS[section_type].format(data)


# =============================================================================
# Public API (backwards compatible)
# =============================================================================


def format_company_section(company_data: dict | None) -> str:
    """Format company data for the prompt."""
    if not company_data:
        return ""

    # Handle company list (from company_search intent)
    if company_data.get("companies"):
        return format_section("companies", company_data)

    # Handle single company
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


def format_activities_section(activities_data: dict | None) -> str:
    """Format activities data for the prompt."""
    return format_section("activities", activities_data) if activities_data else ""


def format_history_section(history_data: dict | None) -> str:
    """Format history data for the prompt."""
    return format_section("history", history_data) if history_data else ""


def format_pipeline_section(pipeline_data: dict | None) -> str:
    """Format pipeline data for the prompt."""
    if not pipeline_data:
        return ""

    summary = pipeline_data.get("summary", {})
    opps = pipeline_data.get("opportunities", [])

    if not summary.get("total_count"):
        return "=== PIPELINE ===\nNo open opportunities."

    lines = [
        "=== PIPELINE SUMMARY ===",
        f"Total Open Deals: {summary.get('total_count', 0)}",
        f"Total Value: ${summary.get('total_value', 0):,.0f}",
        "\nBy Stage:",
    ]

    for stage, data in summary.get("stages", {}).items():
        lines.append(f"  - {stage}: {data.get('count', 0)} deals (${data.get('total_value', 0):,.0f})")

    if opps:
        lines.append("\nOpen Opportunities:")
        for opp in opps[:5]:
            lines.append(_format_opportunity(opp))

    return "\n".join(lines)


def format_renewals_section(renewals_data: dict | None) -> str:
    """Format renewals data with explicit date range."""
    if not renewals_data:
        return ""

    renewals = renewals_data.get("renewals", [])
    if not renewals:
        return "=== UPCOMING RENEWALS ===\nNo renewals in the specified timeframe."

    days = renewals_data.get("days", 90)
    count = renewals_data.get("count", len(renewals))

    today = datetime.now()
    end_date = today + timedelta(days=days)
    date_range = f"{today.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    lines = [f"=== UPCOMING RENEWALS ({count} in next {days} days: {date_range}) ==="]
    for r in renewals[:10]:
        lines.append(_format_renewal(r))

    return "\n".join(lines)


def format_contacts_section(contacts_data: dict | None) -> str:
    """Format contacts search results."""
    return format_section("contacts", contacts_data) if contacts_data else ""


def format_groups_section(groups_data: dict | None) -> str:
    """Format groups data for the prompt."""
    if not groups_data:
        return ""

    # Handle group members list
    members = groups_data.get("members", [])
    if members:
        group_name = groups_data.get("group_name", "Unknown Group")
        lines = [f"=== GROUP: {group_name} ({len(members)} members) ==="]
        for m in members[:10]:
            lines.append(f"- {m.get('name', m.get('company_id', 'N/A'))}")
        return "\n".join(lines)

    # Handle groups list
    groups = groups_data.get("groups", [])
    if groups:
        lines = [f"=== ACCOUNT GROUPS ({len(groups)} groups) ==="]
        for g in groups[:10]:
            lines.append(f"- {g.get('name', 'N/A')} ({g.get('group_id', 'N/A')}): {g.get('description', '')}")
        return "\n".join(lines)

    return ""


def format_attachments_section(attachments_data: dict | None) -> str:
    """Format attachments search results."""
    return format_section("attachments", attachments_data) if attachments_data else ""


def format_docs_section(docs_answer: str) -> str:
    """Format docs RAG answer for the prompt."""
    return f"=== DOCUMENTATION GUIDANCE ===\n{docs_answer}" if docs_answer else ""


def format_account_context_section(account_context: str) -> str:
    """Format account RAG context (notes, attachments) for the prompt."""
    return f"=== ACCOUNT CONTEXT (Notes & Attachments) ===\n{account_context}" if account_context else ""


def format_conversation_history_section(messages: list[dict] | None, max_messages: int = 4) -> str:
    """Format conversation history for the prompt."""
    if not messages:
        return ""

    recent = messages[-max_messages:]
    lines = ["=== RECENT CONVERSATION ==="]
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        if len(content) > 150:
            content = f"{content[:150]}..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


__all__ = [
    "SectionFormatter",
    "FORMATTERS",
    "format_section",
    "format_company_section",
    "format_activities_section",
    "format_history_section",
    "format_pipeline_section",
    "format_renewals_section",
    "format_contacts_section",
    "format_groups_section",
    "format_attachments_section",
    "format_docs_section",
    "format_account_context_section",
    "format_conversation_history_section",
]
