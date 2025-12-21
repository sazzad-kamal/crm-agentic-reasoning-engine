# =============================================================================
# Middleware for Request/Response Processing
# =============================================================================
"""
Custom middleware for logging, timing, request tracking, and rate limiting.
"""

import time
import uuid
import logging
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Handle both package and direct imports
try:
    from backend.config import get_settings
except ImportError:
    from config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimitStore:
    """Simple in-memory rate limit store."""
    
    def __init__(self):
        self.requests: dict[str, list[float]] = defaultdict(list)
    
    def is_rate_limited(self, client_id: str, max_requests: int, window_seconds: int) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        window_start = now - window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= max_requests:
            return True
        
        # Record this request
        self.requests[client_id].append(now)
        return False
    
    def get_remaining(self, client_id: str, max_requests: int, window_seconds: int) -> int:
        """Get remaining requests for client."""
        now = time.time()
        window_start = now - window_seconds
        
        recent_requests = len([
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ])
        
        return max(0, max_requests - recent_requests)


# Global rate limit store
_rate_limit_store = RateLimitStore()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    
    Limits requests per client IP within a time window.
    Returns 429 Too Many Requests when limit exceeded.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        
        # Skip if rate limiting disabled
        if not settings.rate_limit_enabled:
            return await call_next(request)
        
        # Only rate limit API endpoints
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        
        # Skip health check endpoint
        if request.url.path in ["/api/health", "/api/info"]:
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        if _rate_limit_store.is_rate_limited(
            client_ip,
            settings.rate_limit_requests,
            settings.rate_limit_window,
        ):
            remaining = 0
            retry_after = settings.rate_limit_window
            
            logger.warning(
                f"Rate limit exceeded for {client_ip}",
                extra={"client_ip": client_ip}
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": True,
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(settings.rate_limit_requests),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = _rate_limit_store.get_remaining(
            client_ip,
            settings.rate_limit_requests,
            settings.rate_limit_window,
        )
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.
    
    Adds:
    - Request ID header (X-Request-ID)
    - Request timing (X-Response-Time)
    - Structured logging
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        
        # Generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        
        # Store in request state for access in handlers
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request (if enabled)
        if settings.log_requests:
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} - Started",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query": str(request.query_params),
                }
            )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error and re-raise
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} - Error after {elapsed_ms}ms: {e}",
                extra={"request_id": request_id, "error": str(e)},
                exc_info=True,
            )
            raise
        
        # Calculate elapsed time
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
        
        # Log response (if enabled)
        if settings.log_requests:
            log_fn = logger.info if response.status_code < 400 else logger.warning
            log_fn(
                f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({elapsed_ms}ms)",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "elapsed_ms": elapsed_ms,
                }
            )
        
        return response


class CacheControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add cache control headers.
    
    API responses should not be cached by default.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Don't cache API responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        
        return response
