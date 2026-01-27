"""Data models for action quality evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.eval.shared.models import BaseEvalResults

# SLO thresholds for action quality
SLO_ACTION_PASS_RATE = 0.80
SLO_RELEVANCE = 0.90
SLO_ACTIONABILITY = 0.70
SLO_APPROPRIATENESS = 0.85


class ActionCaseResult(BaseModel):
    """Result for a single action evaluation case."""

    question: str
    answer: str
    suggested_action: str | None
    expected_action: bool = False  # Whether an action was expected for this question
    relevance: float = 0.0
    actionability: float = 0.0
    appropriateness: float = 0.0
    action_passed: bool = False  # From judge, or set by runner based on outcome
    explanation: str = ""  # Judge's reasoning (only for judged cases)
    errors: list[str] = Field(default_factory=list)


class ActionEvalResults(BaseEvalResults):
    """Aggregated action evaluation results."""

    cases: list[ActionCaseResult] = Field(default_factory=list)
    avg_relevance: float = 0.0
    avg_actionability: float = 0.0
    avg_appropriateness: float = 0.0

    # Breakdown counts
    action_expected_passed: int = 0  # Action expected, produced, judged pass
    action_expected_failed: int = 0  # Action expected, produced, judged fail
    action_missing: int = 0  # Action expected but not produced
    spurious_action: int = 0  # Action not expected but produced
    correct_silence: int = 0  # Action not expected and not produced
    error_count: int = 0  # Cases that failed due to errors

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual case results."""
        if not self.cases:
            return

        self.passed = sum(1 for c in self.cases if c.action_passed)

        # Breakdown counts (exclude error cases)
        self.error_count = sum(1 for c in self.cases if c.errors)
        self.action_expected_passed = sum(
            1 for c in self.cases if not c.errors and c.expected_action and c.suggested_action and c.action_passed
        )
        self.action_expected_failed = sum(
            1 for c in self.cases if not c.errors and c.expected_action and c.suggested_action and not c.action_passed
        )
        self.action_missing = sum(
            1 for c in self.cases if not c.errors and c.expected_action and not c.suggested_action
        )
        self.spurious_action = sum(
            1 for c in self.cases if not c.errors and not c.expected_action and c.suggested_action
        )
        self.correct_silence = sum(
            1 for c in self.cases if not c.errors and not c.expected_action and not c.suggested_action
        )

        # Action metrics (only for judged cases: expected + produced, no errors)
        judged = [c for c in self.cases if not c.errors and c.suggested_action and c.expected_action]
        if judged:
            self.avg_relevance = sum(c.relevance for c in judged) / len(judged)
            self.avg_actionability = sum(c.actionability for c in judged) / len(judged)
            self.avg_appropriateness = sum(c.appropriateness for c in judged) / len(judged)


__all__ = [
    "ActionCaseResult",
    "ActionEvalResults",
    "SLO_ACTIONABILITY",
    "SLO_ACTION_PASS_RATE",
    "SLO_APPROPRIATENESS",
    "SLO_RELEVANCE",
]
