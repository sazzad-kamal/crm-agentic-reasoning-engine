"""Logging and error suppression utilities for RAGAS evaluation."""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


@contextmanager
def suppress_ragas_logging() -> Generator[None, None, None]:
    """Temporarily suppress RAGAS logging, restoring original level on exit."""
    ragas_logger = logging.getLogger("ragas")
    original_level = ragas_logger.level
    ragas_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        ragas_logger.setLevel(original_level)


def install_event_loop_error_suppression() -> None:
    """Install handlers to suppress harmless 'Event loop is closed' errors.

    These errors occur on Windows with async code during cleanup but don't
    affect functionality. This function suppresses them via:
    - sys.excepthook for uncaught exceptions
    - asyncio exception handler for async errors
    - Log filters for asyncio/httpx/aiohttp/ragas loggers
    """
    import asyncio

    # Suppress via excepthook (catches uncaught exceptions)
    original_excepthook = sys.excepthook

    def custom_excepthook(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
        if isinstance(exc_value, RuntimeError) and "Event loop is closed" in str(exc_value):
            return
        original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = custom_excepthook

    # Suppress via asyncio exception handler
    def silent_exception_handler(loop: Any, context: dict) -> None:
        exception = context.get("exception")
        if isinstance(exception, RuntimeError) and "Event loop is closed" in str(exception):
            return
        loop.default_exception_handler(context)

    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(silent_exception_handler)
    except RuntimeError:
        pass  # No event loop running yet

    # Log filters for noisy loggers
    class EventLoopClosedFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Event loop is closed" not in str(record.getMessage())

    class RagasExecutorFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Exception raised in Job" not in str(record.getMessage())

    for logger_name in ["asyncio", "httpx", "httpcore", "aiohttp"]:
        logging.getLogger(logger_name).addFilter(EventLoopClosedFilter())

    logging.getLogger("ragas.executor").addFilter(RagasExecutorFilter())


__all__ = ["suppress_ragas_logging", "install_event_loop_error_suppression"]
