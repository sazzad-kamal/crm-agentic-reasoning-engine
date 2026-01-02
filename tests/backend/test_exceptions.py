"""
Tests for backend/exceptions.py.

Run with:
    pytest tests/backend/test_exceptions.py -v
"""

import os
import pytest
from fastapi import status

os.environ["MOCK_LLM"] = "1"

from backend.exceptions import (
    ErrorDetail,
    ErrorResponse,
    APIError,
    ValidationError,
    AgentError,
)


# =============================================================================
# ErrorDetail Model Tests
# =============================================================================


class TestErrorDetail:
    """Tests for ErrorDetail model."""

    def test_minimal_creation(self):
        """Test creating ErrorDetail with required fields only."""
        detail = ErrorDetail(code="TEST_ERROR", message="Test message")
        assert detail.code == "TEST_ERROR"
        assert detail.message == "Test message"
        assert detail.field is None
        assert detail.details is None

    def test_full_creation(self):
        """Test creating ErrorDetail with all fields."""
        detail = ErrorDetail(
            code="FIELD_ERROR",
            message="Field validation failed",
            field="username",
            details={"min_length": 3, "max_length": 50},
        )
        assert detail.code == "FIELD_ERROR"
        assert detail.message == "Field validation failed"
        assert detail.field == "username"
        assert detail.details == {"min_length": 3, "max_length": 50}

    def test_model_dump(self):
        """Test that ErrorDetail serializes correctly."""
        detail = ErrorDetail(code="TEST", message="Test")
        dumped = detail.model_dump()
        assert "code" in dumped
        assert "message" in dumped
        assert dumped["code"] == "TEST"

    def test_field_can_be_none(self):
        """Test that field is optional."""
        detail = ErrorDetail(code="ERR", message="msg", field=None)
        assert detail.field is None

    def test_details_accepts_nested_dict(self):
        """Test that details can contain nested structures."""
        nested = {"errors": [{"field": "a", "issue": "b"}], "count": 1}
        detail = ErrorDetail(code="ERR", message="msg", details=nested)
        assert detail.details["errors"][0]["field"] == "a"


# =============================================================================
# ErrorResponse Model Tests
# =============================================================================


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_minimal_creation(self):
        """Test creating ErrorResponse with required fields."""
        response = ErrorResponse(status_code=400, message="Bad request")
        assert response.error is True
        assert response.status_code == 400
        assert response.message == "Bad request"
        assert response.details is None
        assert response.request_id is None

    def test_full_creation(self):
        """Test creating ErrorResponse with all fields."""
        detail = ErrorDetail(code="VAL_ERR", message="Invalid input")
        response = ErrorResponse(
            status_code=422,
            message="Validation failed",
            details=[detail],
            request_id="abc-123",
        )
        assert response.status_code == 422
        assert response.message == "Validation failed"
        assert len(response.details) == 1
        assert response.details[0].code == "VAL_ERR"
        assert response.request_id == "abc-123"

    def test_error_is_always_true(self):
        """Test that error field defaults to True."""
        response = ErrorResponse(status_code=500, message="Error")
        assert response.error is True

    def test_model_dump(self):
        """Test that ErrorResponse serializes correctly."""
        response = ErrorResponse(status_code=404, message="Not found")
        dumped = response.model_dump()
        assert dumped["error"] is True
        assert dumped["status_code"] == 404
        assert dumped["message"] == "Not found"

    def test_multiple_details(self):
        """Test ErrorResponse with multiple error details."""
        details = [
            ErrorDetail(code="ERR1", message="First error"),
            ErrorDetail(code="ERR2", message="Second error"),
        ]
        response = ErrorResponse(status_code=400, message="Multiple errors", details=details)
        assert len(response.details) == 2


# =============================================================================
# APIError Exception Tests
# =============================================================================


class TestAPIError:
    """Tests for APIError exception."""

    def test_basic_creation(self):
        """Test creating basic APIError."""
        error = APIError(status_code=400, message="Bad request")
        assert error.status_code == 400
        assert error.detail == "Bad request"
        assert error.code == "API_ERROR"
        assert error.details is None

    def test_custom_code(self):
        """Test APIError with custom code."""
        error = APIError(status_code=403, message="Forbidden", code="AUTH_ERROR")
        assert error.code == "AUTH_ERROR"

    def test_with_details(self):
        """Test APIError with error details."""
        detail = ErrorDetail(code="DETAIL", message="Detail message")
        error = APIError(status_code=400, message="Error", details=[detail])
        assert error.details is not None
        assert len(error.details) == 1
        assert error.details[0].code == "DETAIL"

    def test_inherits_from_http_exception(self):
        """Test that APIError is an HTTPException."""
        from fastapi import HTTPException

        error = APIError(status_code=500, message="Internal error")
        assert isinstance(error, HTTPException)

    def test_can_be_raised(self):
        """Test that APIError can be raised and caught."""
        with pytest.raises(APIError) as exc_info:
            raise APIError(status_code=404, message="Not found")

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Not found"


# =============================================================================
# ValidationError Exception Tests
# =============================================================================


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_basic_creation(self):
        """Test creating basic ValidationError."""
        error = ValidationError(message="Invalid input")
        assert error.status_code == status.HTTP_400_BAD_REQUEST
        assert error.detail == "Invalid input"
        assert error.code == "VALIDATION_ERROR"

    def test_with_field(self):
        """Test ValidationError with field specified."""
        error = ValidationError(message="Field is required", field="email")
        assert error.details is not None
        assert len(error.details) == 1
        assert error.details[0].field == "email"
        assert error.details[0].code == "VALIDATION_ERROR"

    def test_without_field(self):
        """Test ValidationError without field specified."""
        error = ValidationError(message="General validation error")
        assert error.details[0].field is None

    def test_inherits_from_api_error(self):
        """Test that ValidationError inherits from APIError."""
        error = ValidationError(message="Test")
        assert isinstance(error, APIError)

    def test_status_code_is_400(self):
        """Test that ValidationError always uses 400 status."""
        error = ValidationError(message="Any error")
        assert error.status_code == 400

    def test_can_be_raised(self):
        """Test that ValidationError can be raised."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError(message="Name required", field="name")

        assert exc_info.value.status_code == 400
        assert "Name required" in str(exc_info.value.detail)


# =============================================================================
# AgentError Exception Tests
# =============================================================================


class TestAgentError:
    """Tests for AgentError exception."""

    def test_basic_creation(self):
        """Test creating basic AgentError."""
        error = AgentError(message="Agent failed")
        assert error.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert error.detail == "Agent failed"
        assert error.code == "AGENT_ERROR"

    def test_without_details(self):
        """Test AgentError without details."""
        error = AgentError(message="Error occurred")
        assert error.details is None

    def test_with_details_dict(self):
        """Test AgentError with details dictionary."""
        error = AgentError(
            message="Processing failed", details={"step": "routing", "reason": "timeout"}
        )
        assert error.details is not None
        assert len(error.details) == 1
        assert error.details[0].code == "AGENT_ERROR"
        assert error.details[0].details == {"step": "routing", "reason": "timeout"}

    def test_inherits_from_api_error(self):
        """Test that AgentError inherits from APIError."""
        error = AgentError(message="Test")
        assert isinstance(error, APIError)

    def test_status_code_is_500(self):
        """Test that AgentError always uses 500 status."""
        error = AgentError(message="Any error")
        assert error.status_code == 500

    def test_can_be_raised(self):
        """Test that AgentError can be raised."""
        with pytest.raises(AgentError) as exc_info:
            raise AgentError(message="Agent timeout")

        assert exc_info.value.status_code == 500

    def test_details_none_when_not_provided(self):
        """Test that details is None when not provided."""
        error = AgentError(message="Error", details=None)
        assert error.details is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestExceptionIntegration:
    """Integration tests for exception handling."""

    def test_exception_hierarchy(self):
        """Test that exception hierarchy is correct."""
        from fastapi import HTTPException

        validation_err = ValidationError(message="test")
        agent_err = AgentError(message="test")

        # Both should be APIError
        assert isinstance(validation_err, APIError)
        assert isinstance(agent_err, APIError)

        # Both should be HTTPException
        assert isinstance(validation_err, HTTPException)
        assert isinstance(agent_err, HTTPException)

    def test_different_status_codes(self):
        """Test that different error types have different status codes."""
        validation_err = ValidationError(message="test")
        agent_err = AgentError(message="test")

        assert validation_err.status_code == 400
        assert agent_err.status_code == 500

    def test_error_response_with_api_error(self):
        """Test creating ErrorResponse from APIError attributes."""
        error = APIError(
            status_code=403,
            message="Access denied",
            code="AUTH_FAILED",
            details=[ErrorDetail(code="AUTH", message="Invalid token")],
        )

        response = ErrorResponse(
            status_code=error.status_code,
            message=error.detail,
            details=error.details,
            request_id="req-123",
        )

        assert response.status_code == 403
        assert response.message == "Access denied"
        assert len(response.details) == 1
