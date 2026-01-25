"""Data models for action quality evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class ActionCaseResult(BaseModel):
    """Result for a single action evaluation case."""

    question: str
    answer: str
    suggested_action: str | None
    relevance: float = 0.0
    actionability: float = 0.0
    appropriateness: float = 0.0
    action_passed: bool = False  # From judge (or True if no action)
    errors: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Pass if no errors and action passed (or no action)."""
        if self.errors:
            return False
        return self.action_passed


class ActionEvalResults(BaseModel):
    """Aggregated action evaluation results."""

    total: int = 0
    passed: int = 0
    cases: list[ActionCaseResult] = Field(default_factory=list)
    total_with_actions: int = 0  # Only count cases with actions
    avg_relevance: float = 0.0
    avg_actionability: float = 0.0
    avg_appropriateness: float = 0.0

    @property
    def failed(self) -> int:
        """Number of failed cases."""
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def action_pass_rate(self) -> float:
        """Pass rate for cases with actions only."""
        if self.total_with_actions == 0:
            return 0.0
        action_cases = [c for c in self.cases if c.suggested_action]
        passed_actions = sum(1 for c in action_cases if c.action_passed)
        return passed_actions / self.total_with_actions

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual case results."""
        if not self.cases:
            return

        # Action metrics (only for cases with actions)
        action_cases = [c for c in self.cases if c.suggested_action]
        if action_cases:
            self.avg_relevance = sum(c.relevance for c in action_cases) / len(action_cases)
            self.avg_actionability = sum(c.actionability for c in action_cases) / len(action_cases)
            self.avg_appropriateness = sum(c.appropriateness for c in action_cases) / len(action_cases)


__all__ = ["ActionCaseResult", "ActionEvalResults"]
