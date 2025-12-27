"""Tests for backend exception classes."""

import pytest
from fastapi import status

from backend.exceptions import (
    APIError,
    ValidationError,
    NotFoundError,
    RateLimitError,
    AgentError,
    ErrorDetail,
    ErrorResponse,
)


class TestErrorDetail:
    """Tests for ErrorDetail model."""

    def test_basic_error_detail(self):
        """Test basic ErrorDetail creation."""
        detail = ErrorDetail(code="TEST_ERROR", message="Test message")
        assert detail.code == "TEST_ERROR"
        assert detail.message == "Test message"
        assert detail.field is None
        assert detail.details is None

    def test_error_detail_with_field(self):
        """Test ErrorDetail with field specified."""
        detail = ErrorDetail(code="FIELD_ERROR", message="Invalid value", field="email")
        assert detail.field == "email"

    def test_error_detail_with_details(self):
        """Test ErrorDetail with extra details."""
        extra = {"expected": "string", "got": "int"}
        detail = ErrorDetail(code="TYPE_ERROR", message="Wrong type", details=extra)
        assert detail.details == extra


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_basic_error_response(self):
        """Test basic ErrorResponse creation."""
        response = ErrorResponse(status_code=400, message="Bad request")
        assert response.error is True
        assert response.status_code == 400
        assert response.message == "Bad request"
        assert response.details is None
        assert response.request_id is None

    def test_error_response_with_details(self):
        """Test ErrorResponse with error details."""
        details = [ErrorDetail(code="E1", message="Error 1")]
        response = ErrorResponse(
            status_code=422,
            message="Validation failed",
            details=details,
            request_id="req-123",
        )
        assert len(response.details) == 1
        assert response.request_id == "req-123"


class TestAPIError:
    """Tests for base APIError."""

    def test_api_error_basic(self):
        """Test basic APIError creation."""
        error = APIError(status_code=500, message="Server error")
        assert error.status_code == 500
        assert error.detail == "Server error"
        assert error.code == "API_ERROR"
        assert error.details is None

    def test_api_error_with_custom_code(self):
        """Test APIError with custom code."""
        error = APIError(
            status_code=400,
            message="Custom error",
            code="CUSTOM_CODE",
        )
        assert error.code == "CUSTOM_CODE"

    def test_api_error_is_http_exception(self):
        """Test that APIError inherits from HTTPException."""
        from fastapi import HTTPException
        error = APIError(status_code=400, message="Test")
        assert isinstance(error, HTTPException)


class TestValidationError:
    """Tests for ValidationError."""

    def test_validation_error_basic(self):
        """Test basic ValidationError."""
        error = ValidationError(message="Invalid input")
        assert error.status_code == status.HTTP_400_BAD_REQUEST
        assert error.detail == "Invalid input"
        assert error.code == "VALIDATION_ERROR"

    def test_validation_error_with_field(self):
        """Test ValidationError with field specified."""
        error = ValidationError(message="Email required", field="email")
        assert error.details is not None
        assert len(error.details) == 1
        assert error.details[0].field == "email"


class TestNotFoundError:
    """Tests for NotFoundError."""

    def test_not_found_error(self):
        """Test NotFoundError creation."""
        error = NotFoundError(resource="Contact", identifier="123")
        assert error.status_code == status.HTTP_404_NOT_FOUND
        assert "Contact not found: 123" in error.detail
        assert error.code == "NOT_FOUND"

    def test_not_found_error_different_resource(self):
        """Test NotFoundError with different resource types."""
        error = NotFoundError(resource="Company", identifier="acme-corp")
        assert "Company not found: acme-corp" in error.detail


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_rate_limit_error_default(self):
        """Test RateLimitError with default retry_after."""
        error = RateLimitError()
        assert error.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "60 seconds" in error.detail
        assert error.code == "RATE_LIMIT_EXCEEDED"

    def test_rate_limit_error_custom_retry(self):
        """Test RateLimitError with custom retry_after."""
        error = RateLimitError(retry_after=120)
        assert "120 seconds" in error.detail


class TestAgentError:
    """Tests for AgentError."""

    def test_agent_error_basic(self):
        """Test basic AgentError."""
        error = AgentError(message="Agent failed")
        assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert error.detail == "Agent failed"
        assert error.code == "AGENT_ERROR"
        assert error.details is None

    def test_agent_error_with_details(self):
        """Test AgentError with details dict."""
        details = {"step": "tool_call", "tool": "search"}
        error = AgentError(message="Tool failed", details=details)
        assert error.details is not None
        assert len(error.details) == 1
        assert error.details[0].details == details
