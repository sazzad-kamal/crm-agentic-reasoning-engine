"""
Common utilities and data classes for intent handlers.

Shared by all handler modules to avoid duplication.
"""

import logging
from dataclasses import dataclass, field

from backend.agent.schemas import Source
from backend.agent.tools.company import tool_company_lookup


logger = logging.getLogger(__name__)


@dataclass
class IntentContext:
    """Context passed to intent handlers."""

    question: str
    resolved_company_id: str | None
    days: int
    router_result: object | None = None
    owner: str | None = None  # Role-based owner filter (jsmith, amartin, or None for all)


@dataclass
class IntentResult:
    """Result from an intent handler."""

    raw_data: dict = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    company_data: dict | None = None
    activities_data: dict | None = None
    history_data: dict | None = None
    pipeline_data: dict | None = None
    renewals_data: dict | None = None
    contacts_data: dict | None = None
    groups_data: dict | None = None  # For group-related queries
    attachments_data: dict | None = None
    analytics_data: dict | None = None
    resolved_company_id: str | None = None


def empty_raw_data() -> dict:
    """Create empty raw_data structure."""
    return {
        "companies": [],
        "contacts": [],
        "activities": [],
        "opportunities": [],
        "history": [],
        "renewals": [],
        "attachments": [],
        "pipeline_summary": None,
        "analytics": None,
    }


def safe_extend(target_list: list, source_list: list | None) -> None:
    """Safely extend a list, handling None sources."""
    if source_list:
        target_list.extend(source_list)


def apply_tool_result(
    result: IntentResult,
    tool_result: object,
    data_attr: str,
    raw_data_key: str,
    list_key: str | None = None,
    limit: int = 8,
) -> None:
    """
    Apply a tool result to an IntentResult, handling sources and raw_data.

    Args:
        result: The IntentResult to update
        tool_result: The tool result object with .data and .sources
        data_attr: Attribute name on result (e.g., 'pipeline_data')
        raw_data_key: Key in raw_data dict (e.g., 'opportunities')
        list_key: Key within tool_result.data to extract (e.g., 'opportunities')
                  If None, uses raw_data_key
        limit: Max items to include in raw_data
    """
    setattr(result, data_attr, tool_result.data)
    safe_extend(result.sources, tool_result.sources)

    extract_key = list_key or raw_data_key
    data = tool_result.data.get(extract_key, [])
    if isinstance(data, list):
        result.raw_data[raw_data_key] = data[:limit]
    else:
        result.raw_data[raw_data_key] = data


def lookup_company(result: IntentResult, company_id: str) -> bool:
    """
    Look up a company and add to result if found.

    Returns True if company was found, False otherwise.
    """
    company_result = tool_company_lookup(company_id)
    if company_result.data.get("found"):
        result.company_data = company_result.data
        safe_extend(result.sources, company_result.sources)
        result.raw_data["companies"] = [result.company_data["company"]]
        result.resolved_company_id = result.company_data["company"]["company_id"]
        return True
    result.company_data = company_result.data
    return False


__all__ = [
    "IntentContext",
    "IntentResult",
    "empty_raw_data",
    "safe_extend",
    "apply_tool_result",
    "lookup_company",
    "logger",
]
