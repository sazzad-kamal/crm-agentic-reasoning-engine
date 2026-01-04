"""Custom exceptions for the API."""

from fastapi import HTTPException, status


class APIError(HTTPException):
    """Base API error."""

    def __init__(self, status_code: int, message: str):
        super().__init__(status_code=status_code, detail=message)


class ValidationError(APIError):
    """Request validation error (400)."""

    def __init__(self, message: str, field: str | None = None):
        detail = f"{field}: {message}" if field else message
        super().__init__(status.HTTP_400_BAD_REQUEST, detail)


class AgentError(APIError):
    """Agent pipeline error (500)."""

    def __init__(self, message: str):
        super().__init__(status.HTTP_500_INTERNAL_SERVER_ERROR, message)


__all__ = ["APIError", "ValidationError", "AgentError"]
