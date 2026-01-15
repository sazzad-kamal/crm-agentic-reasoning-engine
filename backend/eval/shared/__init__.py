"""Shared evaluation utilities."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from backend.eval.shared.callback import get_eval_capture, reset_eval_capture, set_eval_data
from backend.eval.shared.formatting import console, print_debug_failures, print_eval_header
from backend.eval.shared.ragas import evaluate_single


def is_mock_mode() -> bool:
    """Check if MOCK_LLM mode is enabled."""
    return os.environ.get("MOCK_LLM", "0") == "1"


def measure_latency_ms(start_time: float) -> float:
    """Calculate latency in milliseconds from start time."""
    return (time.time() - start_time) * 1000


def safe_average(values: list[float], default: float = 0.0) -> float:
    """Calculate average, returning default if list is empty."""
    return sum(values) / len(values) if values else default


def load_project_env() -> None:
    """Load .env from project root."""
    project_root = Path(__file__).parent.parent.parent.parent
    load_dotenv(project_root / ".env")


__all__ = [
    # Callback
    "reset_eval_capture",
    "set_eval_data",
    "get_eval_capture",
    # Formatting
    "console",
    "print_debug_failures",
    "print_eval_header",
    # RAGAS
    "evaluate_single",
    # Utilities
    "is_mock_mode",
    "measure_latency_ms",
    "safe_average",
    "load_project_env",
]
