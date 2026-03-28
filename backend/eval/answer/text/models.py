"""Data models for text quality evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.eval.shared.models import BaseEvalResults

# SLO thresholds for text quality
SLO_TEXT_ANSWER_CORRECTNESS = 0.35  # Semantic match (lenient for phrasing/style)
SLO_TEXT_ANSWER_RELEVANCY = 0.85  # Must address the question asked
SLO_TEXT_FAITHFULNESS = 0.85  # Must be grounded in retrieved data
SLO_TEXT_PASS_RATE = 0.80

# SLO thresholds for 5-dimension LLM judge
SLO_JUDGE_GROUNDING = 0.70
SLO_JUDGE_COMPLETENESS = 0.70
SLO_JUDGE_CLARITY = 0.70
SLO_JUDGE_ACCURACY = 0.80
SLO_JUDGE_ACTIONABILITY = 0.60
SLO_JUDGE_PASS_RATE = 0.80


class TextCaseResult(BaseModel):
    """Result for a single text evaluation case."""

    question: str
    answer: str
    answer_correctness_score: float = 0.0
    answer_relevancy_score: float = 0.0
    faithfulness_score: float = 0.0
    errors: list[str] = Field(default_factory=list)
    # RAGAS reliability tracking
    ragas_metrics_total: int = 0
    ragas_metrics_failed: int = 0
    # 5-dimension LLM judge scores
    judge_grounding: float = 0.0
    judge_completeness: float = 0.0
    judge_clarity: float = 0.0
    judge_accuracy: float = 0.0
    judge_actionability: float = 0.0
    judge_passed: bool = False
    judge_explanation: str = ""

    @property
    def passed(self) -> bool:
        """Pass if no errors and relevancy + faithfulness meet thresholds."""
        if self.errors:
            return False
        return (
            self.answer_relevancy_score >= SLO_TEXT_ANSWER_RELEVANCY
            and self.faithfulness_score >= SLO_TEXT_FAITHFULNESS
        )


class TextEvalResults(BaseEvalResults):
    """Aggregated text evaluation results."""

    cases: list[TextCaseResult] = Field(default_factory=list)
    avg_answer_correctness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_faithfulness: float = 0.0
    # RAGAS reliability tracking
    ragas_metrics_total: int = 0
    ragas_metrics_failed: int = 0
    # 5-dimension LLM judge aggregates
    avg_judge_grounding: float = 0.0
    avg_judge_completeness: float = 0.0
    avg_judge_clarity: float = 0.0
    avg_judge_accuracy: float = 0.0
    avg_judge_actionability: float = 0.0
    judge_pass_count: int = 0

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
        self.avg_faithfulness = sum(c.faithfulness_score for c in self.cases) / n
        self.ragas_metrics_total = sum(c.ragas_metrics_total for c in self.cases)
        self.ragas_metrics_failed = sum(c.ragas_metrics_failed for c in self.cases)
        # Judge aggregates
        judged = [c for c in self.cases if c.judge_grounding > 0 or c.judge_passed]
        if judged:
            nj = len(judged)
            self.avg_judge_grounding = sum(c.judge_grounding for c in judged) / nj
            self.avg_judge_completeness = sum(c.judge_completeness for c in judged) / nj
            self.avg_judge_clarity = sum(c.judge_clarity for c in judged) / nj
            self.avg_judge_accuracy = sum(c.judge_accuracy for c in judged) / nj
            self.avg_judge_actionability = sum(c.judge_actionability for c in judged) / nj
            self.judge_pass_count = sum(1 for c in judged if c.judge_passed)


__all__ = [
    "TextCaseResult",
    "TextEvalResults",
    "SLO_TEXT_ANSWER_CORRECTNESS",
    "SLO_TEXT_ANSWER_RELEVANCY",
    "SLO_TEXT_FAITHFULNESS",
    "SLO_TEXT_PASS_RATE",
    "SLO_JUDGE_GROUNDING",
    "SLO_JUDGE_COMPLETENESS",
    "SLO_JUDGE_CLARITY",
    "SLO_JUDGE_ACCURACY",
    "SLO_JUDGE_ACTIONABILITY",
    "SLO_JUDGE_PASS_RATE",
]
