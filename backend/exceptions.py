# =============================================================================
# Custom Exceptions and Error Handling
# =============================================================================
"""
Standardized error responses for the API.
"""

import sys
from typing import Any, Optional

# Python 3.12+ has override in typing, earlier versions need typing_extensions
if sys.version_info >= (3, 12):
    from typing import override
else:
    try:
        from typing_extensions import override
    except ImportError:
        # Fallback: no-op decorator if typing_extensions not available
        def override(func):
            return func

from fastapi import HTTPException, status
from pydantic import BaseModel


# =============================================================================
# Error Response Models
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: str
    message: str
    field: str | None = None
    details: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: bool = True
    status_code: int
    message: str
    details: Optional[list[ErrorDetail]] = None
    request_id: str | None = None


# =============================================================================
# Custom Exceptions
# =============================================================================

class APIError(HTTPException):
    """Base API error with standardized format."""

    def __init__(
        self,
        status_code: int,
        message: str,
        code: str = "API_ERROR",
        details: Optional[list[ErrorDetail]] = None,
    ):
        self.code = code
        self.details = details
        super().__init__(status_code=status_code, detail=message)


class ValidationError(APIError):
    """Request validation error."""

    @override
    def __init__(self, message: str, field: str | None = None) -> None:
        details = [ErrorDetail(code="VALIDATION_ERROR", message=message, field=field)]
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=message,
            code="VALIDATION_ERROR",
            details=details,
        )


class NotFoundError(APIError):
    """Resource not found error."""

    @override
    def __init__(self, resource: str, identifier: str) -> None:
        message = f"{resource} not found: {identifier}"
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=message,
            code="NOT_FOUND",
        )


class RateLimitError(APIError):
    """Rate limit exceeded."""

    @override
    def __init__(self, retry_after: int = 60) -> None:
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            code="RATE_LIMIT_EXCEEDED",
        )


class AgentError(APIError):
    """Error from the agent pipeline."""

    @override
    def __init__(self, message: str, details: dict | None = None) -> None:
        error_details = [ErrorDetail(code="AGENT_ERROR", message=message, details=details)] if details else None
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            code="AGENT_ERROR",
            details=error_details,
        )


__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "APIError",
    "ValidationError",
    "NotFoundError",
    "RateLimitError",
    "AgentError",
]
