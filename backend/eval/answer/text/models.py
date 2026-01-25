"""Data models for text quality evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field

# SLO thresholds for text quality
SLO_TEXT_ANSWER_CORRECTNESS = 0.70  # Semantic match with expected answer
SLO_TEXT_PASS_RATE = 0.80


class TextCaseResult(BaseModel):
    """Result for a single text evaluation case."""

    question: str
    answer: str
    answer_correctness_score: float = 0.0
    errors: list[str] = Field(default_factory=list)
    # RAGAS reliability tracking
    ragas_metrics_total: int = 0
    ragas_metrics_failed: int = 0

    @property
    def passed(self) -> bool:
        """Pass if no errors and answer correctness meets threshold."""
        if self.errors:
            return False
        return self.answer_correctness_score >= SLO_TEXT_ANSWER_CORRECTNESS


class TextEvalResults(BaseModel):
    """Aggregated text evaluation results."""

    total: int = 0
    passed: int = 0
    cases: list[TextCaseResult] = Field(default_factory=list)
    avg_answer_correctness: float = 0.0
    # RAGAS reliability tracking
    ragas_metrics_total: int = 0
    ragas_metrics_failed: int = 0

    @property
    def failed(self) -> int:
        """Number of failed cases."""
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def ragas_success_rate(self) -> float:
        """RAGAS metrics success rate."""
        if self.ragas_metrics_total == 0:
            return 1.0
        return (self.ragas_metrics_total - self.ragas_metrics_failed) / self.ragas_metrics_total

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual case results."""
        if not self.cases:
            return
        n = len(self.cases)
        self.passed = sum(1 for c in self.cases if c.passed)
        self.avg_answer_correctness = sum(c.answer_correctness_score for c in self.cases) / n
        self.ragas_metrics_total = sum(c.ragas_metrics_total for c in self.cases)
        self.ragas_metrics_failed = sum(c.ragas_metrics_failed for c in self.cases)


__all__ = [
    "TextCaseResult",
    "TextEvalResults",
    "SLO_TEXT_ANSWER_CORRECTNESS",
    "SLO_TEXT_PASS_RATE",
]
