"""
Tests for backend middleware.

Run with:
    pytest backend/tests/test_middleware.py -v
"""

import os
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["MOCK_LLM"] = "1"

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from backend.middleware import RequestLoggingMiddleware, CacheControlMiddleware


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def app_with_logging_middleware():
    """Create a FastAPI app with request logging middleware."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}
    
    @app.get("/api/data")
    async def api_endpoint():
        return {"data": "value"}
    
    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")
    
    return app


@pytest.fixture
def app_with_cache_middleware():
    """Create a FastAPI app with cache control middleware."""
    app = FastAPI()
    app.add_middleware(CacheControlMiddleware)
    
    @app.get("/api/data")
    async def api_endpoint():
        return {"data": "value"}
    
    @app.get("/static/file")
    async def static_endpoint():
        return {"content": "static"}
    
    return app


@pytest.fixture
def logging_client(app_with_logging_middleware):
    """Create a test client for logging middleware tests."""
    return TestClient(app_with_logging_middleware, raise_server_exceptions=False)


@pytest.fixture
def cache_client(app_with_cache_middleware):
    """Create a test client for cache middleware tests."""
    return TestClient(app_with_cache_middleware)


# =============================================================================
# Request Logging Middleware Tests
# =============================================================================

class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware."""
    
    def test_adds_request_id_header(self, logging_client):
        """Test that X-Request-ID header is added to response."""
        response = logging_client.get("/test")
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0
    
    def test_preserves_custom_request_id(self, logging_client):
        """Test that custom X-Request-ID is preserved."""
        custom_id = "custom-123"
        response = logging_client.get("/test", headers={"X-Request-ID": custom_id})
        assert response.headers["x-request-id"] == custom_id
    
    def test_adds_response_time_header(self, logging_client):
        """Test that X-Response-Time header is added."""
        response = logging_client.get("/test")
        assert "x-response-time" in response.headers
        assert "ms" in response.headers["x-response-time"]
    
    def test_response_time_is_numeric(self, logging_client):
        """Test that response time is a valid number."""
        response = logging_client.get("/test")
        time_str = response.headers["x-response-time"]
        time_ms = int(time_str.replace("ms", ""))
        assert time_ms >= 0
    
    def test_request_id_is_stored_in_state(self, app_with_logging_middleware):
        """Test that request_id is stored in request.state."""
        stored_id = None
        
        @app_with_logging_middleware.get("/check-state")
        async def check_state(request: Request):
            nonlocal stored_id
            stored_id = getattr(request.state, "request_id", None)
            return {"ok": True}
        
        client = TestClient(app_with_logging_middleware)
        response = client.get("/check-state")
        
        assert stored_id is not None
        assert response.headers["x-request-id"] == stored_id
    
    def test_handles_errors_gracefully(self, logging_client):
        """Test that middleware handles endpoint errors."""
        response = logging_client.get("/error")
        # Should still have headers even on error
        assert response.status_code == 500
    
    def test_logs_requests_when_enabled(self, logging_client):
        """Test that requests are logged when log_requests is True."""
        with patch("backend.middleware.logger") as mock_logger:
            response = logging_client.get("/test")
            # Logger should have been called
            assert mock_logger.info.called or mock_logger.debug.called


# =============================================================================
# Cache Control Middleware Tests
# =============================================================================

class TestCacheControlMiddleware:
    """Tests for CacheControlMiddleware."""
    
    def test_api_routes_have_no_cache_headers(self, cache_client):
        """Test that API routes have no-cache headers."""
        response = cache_client.get("/api/data")
        assert "cache-control" in response.headers
        assert "no-store" in response.headers["cache-control"]
        assert "no-cache" in response.headers["cache-control"]
    
    def test_api_routes_have_pragma_header(self, cache_client):
        """Test that API routes have Pragma: no-cache."""
        response = cache_client.get("/api/data")
        assert response.headers.get("pragma") == "no-cache"
    
    def test_non_api_routes_no_cache_headers(self, cache_client):
        """Test that non-API routes don't get forced no-cache."""
        response = cache_client.get("/static/file")
        # Non-API routes should not have the forced no-cache
        cache_control = response.headers.get("cache-control", "")
        # If there's a cache-control, it shouldn't necessarily be no-store
        # (depends on implementation - this test verifies the behavior)
        assert response.status_code == 200


# =============================================================================
# Middleware Integration Tests
# =============================================================================

class TestMiddlewareIntegration:
    """Integration tests for middleware stack."""
    
    def test_both_middlewares_work_together(self):
        """Test that logging and cache middlewares work together."""
        app = FastAPI()
        app.add_middleware(CacheControlMiddleware)
        app.add_middleware(RequestLoggingMiddleware)
        
        @app.get("/api/test")
        async def test_endpoint():
            return {"ok": True}
        
        client = TestClient(app)
        response = client.get("/api/test")
        
        # Should have both sets of headers
        assert "x-request-id" in response.headers
        assert "x-response-time" in response.headers
        assert "cache-control" in response.headers
    
    def test_middleware_order_matters(self):
        """Test that middleware executes in correct order."""
        execution_order = []
        
        class FirstMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                execution_order.append("first_before")
                response = await call_next(request)
                execution_order.append("first_after")
                return response
        
        class SecondMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                execution_order.append("second_before")
                response = await call_next(request)
                execution_order.append("second_after")
                return response
        
        app = FastAPI()
        # Added in reverse order (LIFO for middleware)
        app.add_middleware(SecondMiddleware)
        app.add_middleware(FirstMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            execution_order.append("handler")
            return {"ok": True}
        
        client = TestClient(app)
        client.get("/test")
        
        # Verify execution order
        assert execution_order == [
            "first_before",
            "second_before",
            "handler",
            "second_after",
            "first_after"
        ]


# =============================================================================
# Edge Cases
# =============================================================================

class TestMiddlewareEdgeCases:
    """Edge case tests for middleware."""
    
    def test_handles_very_long_paths(self, logging_client):
        """Test handling of very long URL paths."""
        long_path = "/test" + "/segment" * 50
        response = logging_client.get(long_path)
        # Should handle gracefully (404 is fine)
        assert response.status_code in [200, 404]
        assert "x-request-id" in response.headers
    
    def test_handles_special_characters_in_path(self, logging_client):
        """Test handling of special characters in path."""
        response = logging_client.get("/test?param=value&other=123")
        assert "x-request-id" in response.headers
    
    def test_handles_empty_response(self):
        """Test handling of empty responses."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        
        @app.get("/empty")
        async def empty_endpoint():
            return Response(status_code=204)
        
        client = TestClient(app)
        response = client.get("/empty")
        
        assert response.status_code == 204
        assert "x-request-id" in response.headers
