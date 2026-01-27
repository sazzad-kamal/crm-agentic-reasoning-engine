"""Base models shared across all evaluation modules."""

from __future__ import annotations

from pydantic import BaseModel


class BaseEvalResults(BaseModel):
    """Base class for aggregated evaluation results."""

    total: int = 0
    passed: int = 0

    @property
    def failed(self) -> int:
        """Number of failed cases."""
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        return self.passed / self.total if self.total > 0 else 0.0


__all__ = ["BaseEvalResults"]
