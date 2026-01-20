"""Models for followup suggestion evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Question(BaseModel):
    """A question from the evaluation set."""

    text: str
    context: str = ""  # Brief context about what kind of question this is


class CaseResult(BaseModel):
    """Result for a single test case."""

    question: str
    suggestions: list[str] = Field(default_factory=list)
    passed: bool = False
    relevance_score: float = 0.0  # 0-1: Are suggestions relevant?
    diversity_score: float = 0.0  # 0-1: Do suggestions cover different angles?
    errors: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class EvalResults(BaseModel):
    """Aggregated evaluation results."""

    total: int = 0
    passed: int = 0
    cases: list[CaseResult] = Field(default_factory=list)

    # Aggregate metrics
    avg_relevance: float = 0.0
    avg_diversity: float = 0.0
    avg_latency_ms: float = 0.0

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual case results."""
        if not self.cases:
            return

        self.avg_relevance = sum(c.relevance_score for c in self.cases) / len(self.cases)
        self.avg_diversity = sum(c.diversity_score for c in self.cases) / len(self.cases)
        self.avg_latency_ms = sum(c.latency_ms for c in self.cases) / len(self.cases)


__all__ = ["Question", "CaseResult", "EvalResults"]
