"""Data models for followup suggestion evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.eval.shared.models import BaseEvalResults

# SLO thresholds for followup quality
SLO_FOLLOWUP_PASS_RATE = 0.80
SLO_FOLLOWUP_QUESTION_RELEVANCE = 0.60
SLO_FOLLOWUP_ANSWER_GROUNDING = 0.50
SLO_FOLLOWUP_DIVERSITY = 0.50
SLO_FOLLOWUP_ANSWERABILITY = 0.80


class FollowupCaseResult(BaseModel):
    """Result for a single followup evaluation case."""

    question: str
    answer: str = ""
    suggestions: list[str] = Field(default_factory=list)
    passed: bool = False
    question_relevance: float = 0.0
    answer_grounding: float = 0.0
    diversity: float = 0.0
    explanation: str = ""
    answerable_count: int = 0
    answerability: float = 0.0
    errors: list[str] = Field(default_factory=list)


class FollowupEvalResults(BaseEvalResults):
    """Aggregated followup evaluation results."""

    cases: list[FollowupCaseResult] = Field(default_factory=list)
    avg_question_relevance: float = 0.0
    avg_answer_grounding: float = 0.0
    avg_diversity: float = 0.0
    avg_answerability: float = 0.0

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual case results."""
        if not self.cases:
            return

        self.passed = sum(1 for c in self.cases if c.passed)
        self.avg_question_relevance = sum(c.question_relevance for c in self.cases) / len(self.cases)
        self.avg_answer_grounding = sum(c.answer_grounding for c in self.cases) / len(self.cases)
        self.avg_diversity = sum(c.diversity for c in self.cases) / len(self.cases)
        self.avg_answerability = sum(c.answerability for c in self.cases) / len(self.cases)


__all__ = [
    "FollowupCaseResult",
    "FollowupEvalResults",
    "SLO_FOLLOWUP_ANSWERABILITY",
    "SLO_FOLLOWUP_ANSWER_GROUNDING",
    "SLO_FOLLOWUP_DIVERSITY",
    "SLO_FOLLOWUP_PASS_RATE",
    "SLO_FOLLOWUP_QUESTION_RELEVANCE",
]
