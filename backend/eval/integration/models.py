"""Data models for agent evaluation results."""

from pydantic import BaseModel, Field, computed_field

# =============================================================================
# SLO Thresholds
# =============================================================================

SLO_FLOW_PATH_PASS_RATE = 0.85  # 85% of conversation paths should pass


# =============================================================================
# Flow Evaluation Models
# =============================================================================


class FlowStepResult(BaseModel):
    """Result of a single question in a flow."""

    question: str
    answer: str
    latency_ms: int
    has_answer: bool
    # RAGAS metrics (0.0-1.0) - answer quality
    relevance_score: float = 0.0  # RAGAS answer_relevancy
    answer_correctness_score: float = 0.0  # RAGAS answer_correctness
    error: str | None = None
    # RAGAS reliability tracking (per-metric)
    ragas_metrics_total: int = 0  # Number of metrics evaluated (usually 2)
    ragas_metrics_failed: int = 0  # Number of metrics that returned NaN

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Question passes if has answer AND meets quality threshold."""
        return self.has_answer and self.relevance_score >= 0.7


class FlowResult(BaseModel):
    """Result of testing a complete conversation flow."""

    path_id: int
    questions: list[str]
    steps: list[FlowStepResult]
    total_latency_ms: int
    success: bool
    error: str | None = None


class FlowEvalResults(BaseModel):
    """Aggregated results from all flow tests."""

    total_paths: int
    paths_tested: int
    paths_passed: int
    paths_failed: int
    total_questions: int
    questions_passed: int
    questions_failed: int
    # RAGAS metrics (0.0-1.0) - answer quality
    avg_relevance: float = 0.0  # RAGAS answer_relevancy
    avg_answer_correctness: float = 0.0  # RAGAS answer_correctness
    # RAGAS reliability tracking (per-metric, not per-call)
    ragas_metrics_total: int = 0  # Total individual metrics evaluated (questions × 2)
    ragas_metrics_failed: int = 0  # Individual metrics that returned NaN
    # Latency
    total_latency_ms: int = 0
    avg_latency_per_question_ms: float = 0.0
    # Results
    all_results: list[FlowResult] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def path_pass_rate(self) -> float:
        """Percentage of paths that passed."""
        return self.paths_passed / self.paths_tested if self.paths_tested > 0 else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ragas_success_rate(self) -> float:
        """Percentage of RAGAS metrics that succeeded (1.0 = all succeeded, 0.0 = all failed)."""
        if self.ragas_metrics_total == 0:
            return 1.0  # No RAGAS metrics = no failures
        return (self.ragas_metrics_total - self.ragas_metrics_failed) / self.ragas_metrics_total
