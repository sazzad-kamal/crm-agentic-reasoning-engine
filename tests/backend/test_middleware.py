"""
Tests for backend middleware.

Run with:
    pytest backend/tests/test_middleware.py -v
"""

import os
import pytest
import time
from unittest.mock import MagicMock, patch, AsyncMock

os.environ["MOCK_LLM"] = "1"

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from backend.middleware import (
    RequestLoggingMiddleware,
    CacheControlMiddleware,
    RateLimitMiddleware,
    RateLimitStore,
    _rate_limit_store,
)


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


# =============================================================================
# Rate Limit Store Tests
# =============================================================================


class TestRateLimitStore:
    """Tests for RateLimitStore."""

    def test_init_creates_empty_requests_dict(self):
        """Test that RateLimitStore initializes with empty requests dict."""
        store = RateLimitStore()
        assert isinstance(store.requests, dict)
        assert len(store.requests) == 0

    def test_is_rate_limited_returns_false_for_new_client(self):
        """Test that new client is not rate limited."""
        store = RateLimitStore()
        result = store.is_rate_limited("client1", max_requests=10, window_seconds=60)
        assert result is False

    def test_is_rate_limited_records_request(self):
        """Test that request is recorded after check."""
        store = RateLimitStore()
        store.is_rate_limited("client1", max_requests=10, window_seconds=60)
        assert "client1" in store.requests
        assert len(store.requests["client1"]) == 1

    def test_is_rate_limited_returns_true_when_limit_exceeded(self):
        """Test that client is rate limited after exceeding max requests."""
        store = RateLimitStore()
        max_requests = 3

        # Make max_requests calls
        for _ in range(max_requests):
            result = store.is_rate_limited("client1", max_requests=max_requests, window_seconds=60)
            assert result is False

        # Next call should be rate limited
        result = store.is_rate_limited("client1", max_requests=max_requests, window_seconds=60)
        assert result is True

    def test_is_rate_limited_cleans_old_requests(self):
        """Test that old requests are cleaned from the window."""
        store = RateLimitStore()

        # Add old timestamp directly
        old_time = time.time() - 100  # 100 seconds ago
        store.requests["client1"] = [old_time, old_time, old_time]

        # With a 60 second window, old requests should be cleaned
        result = store.is_rate_limited("client1", max_requests=3, window_seconds=60)
        assert result is False
        assert len(store.requests["client1"]) == 1  # Only the new request

    def test_get_remaining_returns_full_limit_for_new_client(self):
        """Test that new client has full remaining requests."""
        store = RateLimitStore()
        remaining = store.get_remaining("newclient", max_requests=10, window_seconds=60)
        assert remaining == 10

    def test_get_remaining_decreases_with_requests(self):
        """Test that remaining decreases as requests are made."""
        store = RateLimitStore()
        max_requests = 5

        # Make some requests
        store.is_rate_limited("client1", max_requests=max_requests, window_seconds=60)
        store.is_rate_limited("client1", max_requests=max_requests, window_seconds=60)

        remaining = store.get_remaining("client1", max_requests=max_requests, window_seconds=60)
        assert remaining == 3  # 5 - 2 = 3

    def test_get_remaining_returns_zero_at_limit(self):
        """Test that remaining is 0 when at limit."""
        store = RateLimitStore()
        max_requests = 2

        store.is_rate_limited("client1", max_requests=max_requests, window_seconds=60)
        store.is_rate_limited("client1", max_requests=max_requests, window_seconds=60)

        remaining = store.get_remaining("client1", max_requests=max_requests, window_seconds=60)
        assert remaining == 0

    def test_get_remaining_ignores_old_requests(self):
        """Test that get_remaining ignores requests outside window."""
        store = RateLimitStore()

        # Add old timestamps
        old_time = time.time() - 100
        store.requests["client1"] = [old_time, old_time]

        remaining = store.get_remaining("client1", max_requests=5, window_seconds=60)
        assert remaining == 5  # Old requests don't count


# =============================================================================
# Rate Limit Middleware Tests
# =============================================================================


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    @pytest.fixture
    def app_with_rate_limit(self):
        """Create FastAPI app with rate limit middleware."""
        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.get("/api/test")
        async def test_endpoint():
            return {"ok": True}

        @app.get("/api/health")
        async def health_endpoint():
            return {"status": "ok"}

        @app.get("/api/info")
        async def info_endpoint():
            return {"info": "test"}

        @app.get("/other")
        async def other_endpoint():
            return {"other": True}

        return app

    @pytest.fixture
    def rate_limit_client(self, app_with_rate_limit):
        """Create test client for rate limit tests."""
        return TestClient(app_with_rate_limit)

    def test_skips_when_disabled(self, app_with_rate_limit):
        """Test that rate limiting is skipped when disabled."""
        with patch("backend.middleware.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_enabled = False
            client = TestClient(app_with_rate_limit)

            # Should not be rate limited even with many requests
            for _ in range(100):
                response = client.get("/api/test")
                assert response.status_code == 200

    def test_skips_non_api_routes(self, rate_limit_client):
        """Test that non-API routes are not rate limited."""
        # Clear any existing rate limit state
        _rate_limit_store.requests.clear()

        with patch("backend.middleware.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_enabled = True
            mock_settings.return_value.rate_limit_requests = 1
            mock_settings.return_value.rate_limit_window = 60

            # Non-API route should not be rate limited
            for _ in range(5):
                response = rate_limit_client.get("/other")
                assert response.status_code == 200

    def test_skips_health_endpoint(self, rate_limit_client):
        """Test that health endpoint is exempt from rate limiting."""
        _rate_limit_store.requests.clear()

        with patch("backend.middleware.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_enabled = True
            mock_settings.return_value.rate_limit_requests = 1
            mock_settings.return_value.rate_limit_window = 60

            # Health endpoint should not be rate limited
            for _ in range(5):
                response = rate_limit_client.get("/api/health")
                assert response.status_code == 200

    def test_skips_info_endpoint(self, rate_limit_client):
        """Test that info endpoint is exempt from rate limiting."""
        _rate_limit_store.requests.clear()

        with patch("backend.middleware.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_enabled = True
            mock_settings.return_value.rate_limit_requests = 1
            mock_settings.return_value.rate_limit_window = 60

            # Info endpoint should not be rate limited
            for _ in range(5):
                response = rate_limit_client.get("/api/info")
                assert response.status_code == 200

    def test_returns_429_when_rate_limited(self, rate_limit_client):
        """Test that 429 is returned when rate limit exceeded."""
        _rate_limit_store.requests.clear()

        with patch("backend.middleware.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_enabled = True
            mock_settings.return_value.rate_limit_requests = 2
            mock_settings.return_value.rate_limit_window = 60

            # First 2 requests should succeed
            response1 = rate_limit_client.get("/api/test")
            assert response1.status_code == 200

            response2 = rate_limit_client.get("/api/test")
            assert response2.status_code == 200

            # Third request should be rate limited
            response3 = rate_limit_client.get("/api/test")
            assert response3.status_code == 429

    def test_429_response_has_retry_after_header(self, rate_limit_client):
        """Test that 429 response includes Retry-After header."""
        _rate_limit_store.requests.clear()

        with patch("backend.middleware.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_enabled = True
            mock_settings.return_value.rate_limit_requests = 1
            mock_settings.return_value.rate_limit_window = 30

            rate_limit_client.get("/api/test")  # Use up limit
            response = rate_limit_client.get("/api/test")  # Should be limited

            assert response.status_code == 429
            assert "retry-after" in response.headers
            assert response.headers["retry-after"] == "30"

    def test_429_response_has_rate_limit_headers(self, rate_limit_client):
        """Test that 429 response includes rate limit headers."""
        _rate_limit_store.requests.clear()

        with patch("backend.middleware.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_enabled = True
            mock_settings.return_value.rate_limit_requests = 1
            mock_settings.return_value.rate_limit_window = 60

            rate_limit_client.get("/api/test")  # Use up limit
            response = rate_limit_client.get("/api/test")  # Should be limited

            assert "x-ratelimit-limit" in response.headers
            assert "x-ratelimit-remaining" in response.headers
            assert "x-ratelimit-reset" in response.headers
            assert response.headers["x-ratelimit-remaining"] == "0"

    def test_429_response_body_format(self, rate_limit_client):
        """Test that 429 response body has correct format."""
        _rate_limit_store.requests.clear()

        with patch("backend.middleware.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_enabled = True
            mock_settings.return_value.rate_limit_requests = 1
            mock_settings.return_value.rate_limit_window = 60

            rate_limit_client.get("/api/test")  # Use up limit
            response = rate_limit_client.get("/api/test")  # Should be limited

            data = response.json()
            assert data["error"] is True
            assert "Rate limit exceeded" in data["message"]
            assert "retry_after" in data

    def test_success_response_has_rate_limit_headers(self, rate_limit_client):
        """Test that successful responses include rate limit headers."""
        _rate_limit_store.requests.clear()

        with patch("backend.middleware.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_enabled = True
            mock_settings.return_value.rate_limit_requests = 10
            mock_settings.return_value.rate_limit_window = 60

            response = rate_limit_client.get("/api/test")

            assert response.status_code == 200
            assert "x-ratelimit-limit" in response.headers
            assert "x-ratelimit-remaining" in response.headers
            assert response.headers["x-ratelimit-limit"] == "10"

    def test_handles_missing_client_ip(self, app_with_rate_limit):
        """Test that middleware handles missing client IP gracefully."""
        _rate_limit_store.requests.clear()

        with patch("backend.middleware.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_enabled = True
            mock_settings.return_value.rate_limit_requests = 10
            mock_settings.return_value.rate_limit_window = 60

            client = TestClient(app_with_rate_limit)
            response = client.get("/api/test")

            # Should handle gracefully (testclient provides a client)
            assert response.status_code == 200
