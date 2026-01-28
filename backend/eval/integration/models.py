"""Data models for integration (flow) evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.eval.shared.models import BaseEvalResults

# SLO thresholds for integration evaluation
SLO_FLOW_PASS_RATE = 0.85  # 85% of conversation paths should pass


class FlowStepResult(BaseModel):
    """Result of a single question in a flow."""

    question: str
    answer: str
    latency_ms: int
    has_answer: bool
    # RAGAS metrics (0.0-1.0) - answer quality
    relevance_score: float = 0.0  # RAGAS answer_relevancy
    answer_correctness_score: float = 0.0  # RAGAS answer_correctness
    errors: list[str] = Field(default_factory=list)
    # RAGAS reliability tracking
    ragas_metrics_total: int = 0  # Number of metrics evaluated (usually 2)
    ragas_metrics_failed: int = 0  # Number of metrics that returned NaN

    @property
    def passed(self) -> bool:
        """Question passes if has answer, no errors, and meets quality threshold."""
        if self.errors:
            return False
        if not self.has_answer:
            return False
        # Only check relevance when RAGAS was actually evaluated
        if self.ragas_metrics_total > 0:
            return self.relevance_score >= 0.7
        return True


class FlowResult(BaseModel):
    """Result of testing a complete conversation flow."""

    path_id: int
    questions: list[str]
    steps: list[FlowStepResult]
    total_latency_ms: int
    success: bool


class FlowEvalResults(BaseEvalResults):
    """Aggregated results from all flow tests."""

    cases: list[FlowResult] = Field(default_factory=list)
    # RAGAS metrics (0.0-1.0) - answer quality
    avg_relevance: float = 0.0  # RAGAS answer_relevancy
    avg_answer_correctness: float = 0.0  # RAGAS answer_correctness
    # RAGAS reliability tracking
    ragas_metrics_total: int = 0  # Total individual metrics evaluated (questions × 2)
    ragas_metrics_failed: int = 0  # Individual metrics that returned NaN
    # Latency
    total_latency_ms: int = 0
    avg_latency_per_question_ms: float = 0.0

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

        self.passed = sum(1 for c in self.cases if c.success)

        all_steps = [s for c in self.cases for s in c.steps]
        if all_steps:
            n = len(all_steps)
            self.avg_relevance = sum(s.relevance_score for s in all_steps) / n
            self.avg_answer_correctness = sum(s.answer_correctness_score for s in all_steps) / n
            self.ragas_metrics_total = sum(s.ragas_metrics_total for s in all_steps)
            self.ragas_metrics_failed = sum(s.ragas_metrics_failed for s in all_steps)

        self.total_latency_ms = sum(c.total_latency_ms for c in self.cases)
        total_questions = sum(len(c.steps) for c in self.cases)
        self.avg_latency_per_question_ms = (
            self.total_latency_ms / total_questions if total_questions > 0 else 0.0
        )
