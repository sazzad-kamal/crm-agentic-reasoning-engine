"""Data models for agent evaluation results."""

from pydantic import BaseModel, Field, computed_field

# =============================================================================
# SLO Thresholds
# =============================================================================

# Flow Eval SLOs
SLO_FLOW_PATH_PASS_RATE = 0.85  # 85% of conversation paths should pass
SLO_FLOW_QUESTION_PASS_RATE = 0.90  # 90% of individual questions should pass
SLO_FLOW_FAITHFULNESS = 0.90  # 90% - critical for CRM, no hallucination allowed
SLO_FLOW_RELEVANCE = 0.85  # 85% - answers should address the question
SLO_FLOW_ANSWER_CORRECTNESS = 0.70  # 70% - hardest metric, flexible answer formats
SLO_FLOW_AVG_LATENCY_MS = 4000  # 4s average per question


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
    faithfulness_score: float = 0.0  # RAGAS faithfulness
    answer_correctness_score: float = 0.0  # RAGAS answer_correctness
    judge_explanation: str = ""
    error: str | None = None
    # RAGAS reliability tracking (per-metric)
    ragas_metrics_total: int = 0  # Number of metrics evaluated (usually 3)
    ragas_metrics_failed: int = 0  # Number of metrics that returned NaN

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Question passes if has answer AND meets quality thresholds."""
        return (
            self.has_answer
            and self.relevance_score >= 0.7
            and self.faithfulness_score >= 0.7
        )


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
    avg_faithfulness: float = 0.0  # RAGAS faithfulness
    avg_answer_correctness: float = 0.0  # RAGAS answer_correctness
    # RAGAS reliability tracking (per-metric, not per-call)
    ragas_metrics_total: int = 0  # Total individual metrics evaluated (questions × 3)
    ragas_metrics_failed: int = 0  # Individual metrics that returned NaN
    # Latency
    total_latency_ms: int = 0
    avg_latency_per_question_ms: float = 0.0
    wall_clock_ms: int = 0  # Total wall-clock time for the eval
    # Results
    failed_paths: list[FlowResult] = Field(default_factory=list)
    all_results: list[FlowResult] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def path_pass_rate(self) -> float:
        """Percentage of paths that passed."""
        return self.paths_passed / self.paths_tested if self.paths_tested > 0 else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def question_pass_rate(self) -> float:
        """Percentage of questions that passed."""
        return self.questions_passed / self.total_questions if self.total_questions > 0 else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ragas_success_rate(self) -> float:
        """Percentage of RAGAS metrics that succeeded (1.0 = all succeeded, 0.0 = all failed)."""
        if self.ragas_metrics_total == 0:
            return 1.0  # No RAGAS metrics = no failures
        return (self.ragas_metrics_total - self.ragas_metrics_failed) / self.ragas_metrics_total

