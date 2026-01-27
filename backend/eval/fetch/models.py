"""Models for fetch node evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.eval.shared.models import BaseEvalResults


class Question(BaseModel):
    """A question from the evaluation set."""

    text: str
    difficulty: int = 1
    expected_sql: str | None = None  # Optional expected SQL for semantic comparison


class CaseResult(BaseModel):
    """Result for a single test case."""

    question: Question
    sql: str
    passed: bool
    errors: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class EvalResults(BaseEvalResults):
    """Aggregated evaluation results."""

    cases: list[CaseResult] = Field(default_factory=list)
    avg_latency_ms: float = 0.0

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual case results."""
        if self.cases:
            self.avg_latency_ms = sum(c.latency_ms for c in self.cases) / len(self.cases)


__all__ = ["Question", "CaseResult", "EvalResults"]
