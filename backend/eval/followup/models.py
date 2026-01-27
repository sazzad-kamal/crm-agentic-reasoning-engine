"""Data models for followup suggestion evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field

# SLO thresholds for followup quality
SLO_FOLLOWUP_PASS_RATE = 0.80
SLO_FOLLOWUP_RELEVANCE = 0.60
SLO_FOLLOWUP_DIVERSITY = 0.50


class FollowupCaseResult(BaseModel):
    """Result for a single followup evaluation case."""

    question: str
    suggestions: list[str] = Field(default_factory=list)
    passed: bool = False
    relevance: float = 0.0
    diversity: float = 0.0
    explanation: str = ""
    errors: list[str] = Field(default_factory=list)


class FollowupEvalResults(BaseModel):
    """Aggregated followup evaluation results."""

    total: int = 0
    passed: int = 0
    cases: list[FollowupCaseResult] = Field(default_factory=list)
    avg_relevance: float = 0.0
    avg_diversity: float = 0.0

    @property
    def failed(self) -> int:
        """Number of failed cases."""
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        return self.passed / self.total if self.total > 0 else 0.0

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual case results."""
        if not self.cases:
            return

        self.passed = sum(1 for c in self.cases if c.passed)
        self.avg_relevance = sum(c.relevance for c in self.cases) / len(self.cases)
        self.avg_diversity = sum(c.diversity for c in self.cases) / len(self.cases)


__all__ = [
    "FollowupCaseResult",
    "FollowupEvalResults",
    "SLO_FOLLOWUP_DIVERSITY",
    "SLO_FOLLOWUP_PASS_RATE",
    "SLO_FOLLOWUP_RELEVANCE",
]
