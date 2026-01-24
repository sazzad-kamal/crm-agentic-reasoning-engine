"""Data models for answer evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class Question(BaseModel):
    """A question from the evaluation set."""

    text: str
    difficulty: int = 1
    expected_sql: str
    expected_answer: str = ""  # For RAGAS answer_correctness (empty = skip metric)


class CaseResult(BaseModel):
    """Result for a single test case."""

    question: str
    answer: str
    suggested_action: str | None
    latency_ms: int
    # RAGAS metrics
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    answer_correctness_score: float = 0.0
    # Action metrics
    action_relevance: float = 0.0
    action_actionability: float = 0.0
    action_appropriateness: float = 0.0
    action_passed: bool = False
    # Errors
    errors: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Pass if RAGAS scores are good AND action passes (if present)."""
        ragas_ok = self.faithfulness_score >= 0.6 and self.relevance_score >= 0.6
        action_ok = self.action_passed if self.suggested_action else True
        return ragas_ok and action_ok and not self.errors


class EvalResults(BaseModel):
    """Aggregated evaluation results."""

    total: int = 0
    passed: int = 0
    cases: list[CaseResult] = Field(default_factory=list)
    # RAGAS averages
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_answer_correctness: float = 0.0
    ragas_pass_rate: float = 0.0
    # Action averages
    avg_action_relevance: float = 0.0
    avg_action_actionability: float = 0.0
    avg_action_appropriateness: float = 0.0
    action_pass_rate: float = 0.0
    # Latency
    avg_latency_ms: float = 0.0

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
        self.avg_latency_ms = sum(c.latency_ms for c in self.cases) / n
        self.avg_faithfulness = sum(c.faithfulness_score for c in self.cases) / n
        self.avg_relevance = sum(c.relevance_score for c in self.cases) / n
        self.avg_answer_correctness = sum(c.answer_correctness_score for c in self.cases) / n
        self.ragas_pass_rate = (
            sum(1 for c in self.cases if c.faithfulness_score >= 0.6 and c.relevance_score >= 0.6) / n
        )
        # Action metrics (only for cases with actions)
        action_cases = [c for c in self.cases if c.suggested_action]
        if action_cases:
            self.avg_action_relevance = sum(c.action_relevance for c in action_cases) / len(action_cases)
            self.avg_action_actionability = sum(c.action_actionability for c in action_cases) / len(action_cases)
            self.avg_action_appropriateness = sum(c.action_appropriateness for c in action_cases) / len(action_cases)
            self.action_pass_rate = sum(1 for c in action_cases if c.action_passed) / len(action_cases)


__all__ = ["Question", "CaseResult", "EvalResults"]
