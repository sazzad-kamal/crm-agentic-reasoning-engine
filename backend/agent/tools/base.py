"""
Base utilities for CRM tools.

Provides the crm_tool decorator and helper functions.
"""

from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from backend.agent.datastore import get_datastore, CRMDataStore
from backend.agent.schemas import Source, ToolResult

P = ParamSpec("P")
T = TypeVar("T")


def make_sources(
    data: list | dict | None,
    source_type: str,
    source_id: str,
    label: str,
) -> list[Source]:
    """Create a source list if data is non-empty."""
    if data:
        return [Source(type=source_type, id=source_id, label=label)]
    return []


def with_datastore(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """
    Decorator that injects datastore if not provided.

    Allows tools to accept an optional `datastore` parameter for testing,
    while using the singleton in production.
    """
    @wraps(func)
    def wrapper(*args, datastore: CRMDataStore | None = None, **kwargs) -> ToolResult:
        ds = datastore or get_datastore()
        return func(*args, datastore=ds, **kwargs)
    return wrapper


__all__ = [
    "make_sources",
    "with_datastore",
    "ToolResult",
    "Source",
    "CRMDataStore",
]
