# =============================================================================
# Middleware for Request/Response Processing
# =============================================================================
"""
Custom middleware for logging, timing, and request tracking.
"""

import time
import uuid
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Handle both package and direct imports
try:
    from backend.config import get_settings
except ImportError:
    from config import get_settings

logger = logging.getLogger(__name__)


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
