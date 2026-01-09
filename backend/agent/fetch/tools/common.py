"""
Common utilities and data classes for intent handlers.

Shared by all handler modules to avoid duplication.
Includes tool helpers merged from tools/base.py.
"""

import json
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any

from backend.agent.core.state import Source
from backend.agent.datastore import CRMDataStore, get_datastore
from backend.agent.fetch.tools.schemas import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Helpers (merged from tools/base.py)
# =============================================================================


def make_sources(
    data: list | dict | None,
    source_type: str,
    source_id: str,
    label: str,
) -> list[Source]:
    """Create a Source list if data exists."""
    if data:
        return [Source(type=source_type, id=source_id, label=label)]
    return []


def with_datastore(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """Decorator that injects a datastore instance if not provided."""
    @wraps(func)
    def wrapper(*args: Any, datastore: CRMDataStore | None = None, **kwargs: Any) -> ToolResult:
        ds = datastore or get_datastore()
        return func(*args, datastore=ds, **kwargs)
    return wrapper


@with_datastore
def tool_company_lookup(
    company_id_or_name: str, datastore: CRMDataStore | None = None
) -> ToolResult:
    """Look up company information by ID or name."""
    ds = datastore
    assert ds is not None  # Guaranteed by @with_datastore

    company_id = ds.resolve_company_id(company_id_or_name)

    if not company_id:
        matches = ds.get_company_name_matches(company_id_or_name, limit=5)
        return ToolResult(
            data={"found": False, "query": company_id_or_name, "close_matches": matches},
            sources=[],
            error=f"Company '{company_id_or_name}' not found",
        )

    company = ds.get_company(company_id)
    if not company:
        return ToolResult(
            data={"found": False, "query": company_id_or_name},
            sources=[],
            error=f"Company '{company_id}' not found in database",
        )

    contacts = ds.get_contacts_for_company(company_id, limit=5)

    return ToolResult(
        data={"found": True, "company": company, "contacts": contacts},
        sources=[Source(type="company", id=company_id, label=company.get("name", company_id))],
    )


@dataclass
class IntentContext:
    """Context passed to intent handlers."""

    question: str
    resolved_company_id: str | None
    days: int
    owner: str | None = None  # Role-based owner filter (jsmith, amartin, or None for all)
    router_result: Any | None = None  # Optional router result for advanced routing


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
    tool_result: ToolResult,
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
    tool_result = tool_company_lookup(company_id)
    result.company_data = tool_result.data
    result.sources.extend(tool_result.sources)

    if not tool_result.data.get("found"):
        return False

    company = tool_result.data["company"]
    result.raw_data["companies"] = [company]
    result.resolved_company_id = company.get("company_id")
    return True


# =============================================================================
# Nested Data Enrichment
# =============================================================================


def _get_csv_path() -> Path:
    """Get the path to the CSV data directory."""
    return Path(__file__).parent.parent.parent / "data" / "csv"


@lru_cache(maxsize=1)
def _load_private_texts() -> dict[str, list[dict[str, Any]]]:
    """Load and cache private_texts.jsonl grouped by company_id."""
    jsonl_path = _get_csv_path() / "private_texts.jsonl"
    if not jsonl_path.exists():
        return {}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                company_id = record.get("company_id", "")
                if company_id:
                    grouped[company_id].append(record)
            except json.JSONDecodeError:
                continue
    return dict(grouped)


@lru_cache(maxsize=1)
def _load_attachments() -> dict[str, list[dict[str, Any]]]:
    """Load and cache attachments.csv grouped by opportunity_id."""
    import csv

    csv_path = _get_csv_path() / "attachments.csv"
    if not csv_path.exists():
        return {}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opp_id = row.get("opportunity_id", "")
            if opp_id:
                grouped[opp_id].append(dict(row))
    return dict(grouped)


def enrich_raw_data(raw_data: dict) -> dict:
    """
    Enrich raw_data with nested fields for display in DataTables.

    Adds:
    - _private_texts to companies (from private_texts.jsonl)
    - _attachments to opportunities (from attachments.csv)
    """
    private_texts = _load_private_texts()
    attachments = _load_attachments()

    # Enrich companies with _private_texts
    for company in raw_data.get("companies", []):
        company_id = company.get("company_id", "")
        company["_private_texts"] = private_texts.get(company_id, [])

    # Enrich opportunities with _attachments
    for opp in raw_data.get("opportunities", []):
        opp_id = opp.get("opportunity_id", "")
        opp["_attachments"] = attachments.get(opp_id, [])

    return raw_data


__all__ = [
    "IntentContext",
    "IntentResult",
    "empty_raw_data",
    "safe_extend",
    "apply_tool_result",
    "lookup_company",
    "enrich_raw_data",
    "make_sources",
    "with_datastore",
    "tool_company_lookup",
    "ToolResult",
    "CRMDataStore",
    "get_datastore",
    "Source",
    "logger",
]
