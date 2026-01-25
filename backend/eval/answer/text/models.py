"""Data models for text quality evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class TextCaseResult(BaseModel):
    """Result for a single text evaluation case."""

    question: str
    answer: str
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    answer_correctness_score: float = 0.0
    errors: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Pass if no errors and RAGAS scores meet thresholds."""
        if self.errors:
            return False
        return self.faithfulness_score >= 0.6 and self.relevance_score >= 0.6


class TextEvalResults(BaseModel):
    """Aggregated text evaluation results."""

    total: int = 0
    passed: int = 0
    cases: list[TextCaseResult] = Field(default_factory=list)
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_answer_correctness: float = 0.0

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
        n = len(self.cases)
        self.avg_faithfulness = sum(c.faithfulness_score for c in self.cases) / n
        self.avg_relevance = sum(c.relevance_score for c in self.cases) / n
        self.avg_answer_correctness = sum(c.answer_correctness_score for c in self.cases) / n


__all__ = ["TextCaseResult", "TextEvalResults"]
