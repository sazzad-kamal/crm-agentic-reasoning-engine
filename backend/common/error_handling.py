"""
Error handling utilities for pipeline nodes and functions.

Consolidates the repetitive try/except patterns into reusable decorators.
"""

import logging
import functools
from typing import Callable, TypeVar, Any


logger = logging.getLogger(__name__)

T = TypeVar("T")


def pipeline_node(
    step_id: str,
    step_label: str,
    error_message: str = "Step failed",
) -> Callable[[Callable[..., dict]], Callable[..., dict]]:
    """
    Decorator for pipeline node functions with standardized error handling.
    
    Wraps a function that returns a dict and adds:
    - Automatic error catching and logging
    - Standardized error response format with steps
    
    Args:
        step_id: Unique identifier for this step
        step_label: Human-readable label for the step
        error_message: Error message prefix
        
    Usage:
        @pipeline_node("fetch_data", "Fetching data", "Data fetch failed")
        def fetch_data_node(state: dict) -> dict:
            # ... logic ...
            return {"data": result, "steps": [...]}
    """
    def decorator(func: Callable[..., dict]) -> Callable[..., dict]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> dict:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[{step_id}] {error_message}: {e}")
                return {
                    "error": f"{error_message}: {str(e)}",
                    "steps": [
                        {"id": step_id, "label": step_label, "status": "error"}
                    ],
                }
        return wrapper
    return decorator


def safe_operation(
    default: T = None,
    log_errors: bool = True,
    error_message: str = "Operation failed",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for safe execution with default fallback.
    
    Useful for non-critical operations that shouldn't break the pipeline.
    
    Args:
        default: Default value to return on error
        log_errors: Whether to log errors
        error_message: Error message to log
        
    Usage:
        @safe_operation(default="", error_message="Query rewrite failed")
        def rewrite_query(query: str) -> str:
            # ... logic that might fail ...
            return rewritten
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"{error_message}: {e}")
                return default
        return wrapper
    return decorator


class PipelineError(Exception):
    """Custom exception for pipeline errors with step tracking."""
    
    def __init__(
        self,
        message: str,
        step_id: str = "unknown",
        step_label: str = "Unknown step",
        original_error: Exception = None,
    ):
        super().__init__(message)
        self.step_id = step_id
        self.step_label = step_label
        self.original_error = original_error
    
    def to_response(self) -> dict:
        """Convert to standard error response dict."""
        return {
            "error": str(self),
            "steps": [
                {"id": self.step_id, "label": self.step_label, "status": "error"}
            ],
        }
