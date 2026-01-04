"""
Tests for backend/core/exceptions.py.

Run with:
    pytest tests/backend/test_exceptions.py -v
"""

import os
import pytest
from fastapi import status

os.environ["MOCK_LLM"] = "1"

from backend.core.exceptions import (
    ErrorResponse,
    APIError,
    ValidationError,
    AgentError,
)


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
        assert response.request_id is None

    def test_with_request_id(self):
        """Test creating ErrorResponse with request_id."""
        response = ErrorResponse(
            status_code=422,
            message="Validation failed",
            request_id="abc-123",
        )
        assert response.status_code == 422
        assert response.message == "Validation failed"
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

    def test_with_field(self):
        """Test ValidationError with field specified."""
        error = ValidationError(message="Field is required", field="email")
        assert error.status_code == 400
        assert "email" in error.detail
        assert "Field is required" in error.detail

    def test_without_field(self):
        """Test ValidationError without field specified."""
        error = ValidationError(message="General validation error")
        assert error.detail == "General validation error"

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

    def test_error_response_from_api_error(self):
        """Test creating ErrorResponse from APIError attributes."""
        error = APIError(status_code=403, message="Access denied")

        response = ErrorResponse(
            status_code=error.status_code,
            message=error.detail,
            request_id="req-123",
        )

        assert response.status_code == 403
        assert response.message == "Access denied"
        assert response.request_id == "req-123"
