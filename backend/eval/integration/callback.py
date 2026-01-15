"""Context-based capture for eval-specific data from graph execution.

Uses contextvars to capture eval data out-of-band, keeping workflow state clean.
This is thread-safe and works with concurrent eval execution.
"""

from contextvars import ContextVar
from typing import Any

# Context variable for eval data capture (thread-safe)
_eval_capture: ContextVar[dict[str, Any] | None] = ContextVar("eval_capture", default=None)


def reset_eval_capture() -> None:
    """Reset the eval capture before graph execution."""
    _eval_capture.set({
        "sql_plan": None,
        "sql_queries_total": 0,
        "sql_queries_success": 0,
        "account_rag_invoked": False,
        "account_chunks": [],
    })


def set_eval_data(**kwargs: Any) -> None:
    """Set eval data from within a node (called by fetch_node)."""
    current = _eval_capture.get()
    if current is None:
        current = {}
    current.update(kwargs)
    _eval_capture.set(current)


def get_eval_capture() -> dict[str, Any]:
    """Get captured eval data after graph execution."""
    current = _eval_capture.get()
    if current is None:
        return {
            "sql_plan": None,
            "sql_queries_total": 0,
            "sql_queries_success": 0,
            "account_rag_invoked": False,
            "account_chunks": [],
        }
    return current.copy()


__all__ = ["reset_eval_capture", "set_eval_data", "get_eval_capture"]
