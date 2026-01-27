"""Data models for text quality evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.eval.shared.models import BaseEvalResults

# SLO thresholds for text quality
SLO_TEXT_ANSWER_CORRECTNESS = 0.50  # Semantic match (lenient for phrasing)
SLO_TEXT_ANSWER_RELEVANCY = 0.85  # Must address the question asked
SLO_TEXT_PASS_RATE = 0.80


class TextCaseResult(BaseModel):
    """Result for a single text evaluation case."""

    question: str
    answer: str
    answer_correctness_score: float = 0.0
    answer_relevancy_score: float = 0.0
    errors: list[str] = Field(default_factory=list)
    # RAGAS reliability tracking
    ragas_metrics_total: int = 0
    ragas_metrics_failed: int = 0

    @property
    def passed(self) -> bool:
        """Pass if no errors and both correctness and relevancy meet thresholds."""
        if self.errors:
            return False
        return (
            self.answer_correctness_score >= SLO_TEXT_ANSWER_CORRECTNESS
            and self.answer_relevancy_score >= SLO_TEXT_ANSWER_RELEVANCY
        )


class TextEvalResults(BaseEvalResults):
    """Aggregated text evaluation results."""

    cases: list[TextCaseResult] = Field(default_factory=list)
    avg_answer_correctness: float = 0.0
    avg_answer_relevancy: float = 0.0
    # RAGAS reliability tracking
    ragas_metrics_total: int = 0
    ragas_metrics_failed: int = 0

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
        self.avg_answer_relevancy = sum(c.answer_relevancy_score for c in self.cases) / n
        self.ragas_metrics_total = sum(c.ragas_metrics_total for c in self.cases)
        self.ragas_metrics_failed = sum(c.ragas_metrics_failed for c in self.cases)


__all__ = [
    "TextCaseResult",
    "TextEvalResults",
    "SLO_TEXT_ANSWER_CORRECTNESS",
    "SLO_TEXT_ANSWER_RELEVANCY",
    "SLO_TEXT_PASS_RATE",
]
