"""
Request logging middleware for tracking and debugging.
"""

import time
import uuid
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core.config import get_settings

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request tracking and timing.

    Adds:
    - Request ID header (X-Request-ID)
    - Response timing (X-Response-Time)
    - Structured logging
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging, timing, and request ID tracking."""
        settings = get_settings()

        # Generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()

        # Log request start
        if settings.log_requests:
            logger.info(
                f"[{request_id}] {request.method} {request.url.path}",
                extra={"request_id": request_id, "method": request.method, "path": request.url.path},
            )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} - Error after {elapsed_ms}ms: {e}",
                extra={"request_id": request_id},
                exc_info=True,
            )
            raise

        # Add headers
        elapsed_ms = int((time.time() - start_time) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"

        # Log response
        if settings.log_requests:
            log_fn = logger.info if response.status_code < 400 else logger.warning
            log_fn(
                f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({elapsed_ms}ms)",
                extra={"request_id": request_id, "status_code": response.status_code, "elapsed_ms": elapsed_ms},
            )

        return response


__all__ = ["RequestLoggingMiddleware"]
