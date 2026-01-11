"""
Data models for agent evaluation results.
"""

from dataclasses import dataclass, field

# =============================================================================
# Composite Score Helpers
# =============================================================================


def _latency_score(avg_latency_ms: float, slo_target_ms: float) -> float:
    """
    Convert latency to 0-1 score for composite score calculation.

    - Score = 1.0 if avg_latency <= SLO target
    - Score = 0.0 if avg_latency >= 2x SLO target
    - Linear interpolation between
    """
    if avg_latency_ms <= slo_target_ms:
        return 1.0
    if avg_latency_ms >= 2 * slo_target_ms:
        return 0.0
    return 1.0 - (avg_latency_ms - slo_target_ms) / slo_target_ms


# =============================================================================
# SLO Thresholds
# =============================================================================

# Latency % SLOs (percentage of total latency per node)
SLO_LATENCY_ROUTING_PCT = 0.25  # 25% - routing should be quick
SLO_LATENCY_RETRIEVAL_PCT = 0.35  # 35% - RAG fetch from vector DB
SLO_LATENCY_ANSWER_PCT = 0.30  # 30% - LLM generation

# Routing Quality SLOs
SLO_SQL_SUCCESS = 0.95  # 95% SQL query generation success rate
SLO_SQL_DATA = 0.90  # 90% SQL data validation success rate

# Flow Eval SLOs
SLO_FLOW_PATH_PASS_RATE = 0.85  # 85% of conversation paths should pass
SLO_FLOW_QUESTION_PASS_RATE = 0.90  # 90% of individual questions should pass
SLO_FLOW_FAITHFULNESS = 0.90  # 90% - critical for CRM, no hallucination allowed
SLO_FLOW_RELEVANCE = 0.85  # 85% - answers should address the question
SLO_FLOW_ANSWER_CORRECTNESS = 0.70  # 70% - hardest metric, flexible answer formats
SLO_FLOW_AVG_LATENCY_MS = 4000  # 4s average per question
SLO_FLOW_COMPOSITE_SCORE = 0.85  # 85% - Flow eval composite

# RAG Retrieval Quality SLOs (account RAG only)
SLO_ACCOUNT_PRECISION = 0.80  # 80% - account retrieval precision
SLO_ACCOUNT_RECALL = 0.70  # 70% - account retrieval recall


# =============================================================================
# Flow Evaluation Models
# =============================================================================


@dataclass
class FlowStepResult:
    """Result of a single question in a flow."""

    question: str
    answer: str
    latency_ms: int
    has_answer: bool
    has_sources: bool
    # SQL execution metrics
    sql_queries_total: int = 0
    sql_queries_success: int = 0
    # SQL data validation (compares results against expected)
    sql_data_validated: bool | None = None  # None = no expected results, True = passed, False = failed
    sql_data_errors: list[str] | None = None  # List of validation failures
    # RAGAS metrics (0.0-1.0) - answer quality
    relevance_score: float = 0.0  # RAGAS answer_relevancy
    faithfulness_score: float = 0.0  # RAGAS faithfulness
    answer_correctness_score: float = 0.0  # RAGAS answer_correctness
    # RAGAS metrics (0.0-1.0) - retrieval quality
    account_precision_score: float = 0.0  # Account RAG precision
    account_recall_score: float = 0.0  # Account RAG recall
    # Per-metric success flags (for excluding NaN from averages)
    precision_succeeded: bool = False  # True if precision metric returned valid value
    recall_succeeded: bool = False  # True if recall metric returned valid value
    # RAG invocation flags (for N/A vs 0% distinction)
    account_rag_invoked: bool = False  # True only if account RAG was called
    # RAG detection accuracy (needs_rag decision from slot planner vs expected)
    rag_decision_correct: bool | None = None  # True if matches expected, False if mismatch, None if no expectation
    judge_explanation: str = ""
    error: str | None = None
    # RAGAS reliability tracking (per-metric)
    ragas_metrics_total: int = 0  # Number of metrics evaluated (usually 5)
    ragas_metrics_failed: int = 0  # Number of metrics that returned NaN

    @property
    def passed(self) -> bool:
        """Question passes if has answer AND meets quality thresholds."""
        return (
            self.has_answer
            and self.relevance_score >= 0.7
            and self.faithfulness_score >= 0.7
        )


@dataclass
class FlowResult:
    """Result of testing a complete conversation flow."""

    path_id: int
    questions: list[str]
    steps: list[FlowStepResult]
    total_latency_ms: int
    success: bool
    error: str | None = None


@dataclass
class FlowEvalResults:
    """Aggregated results from all flow tests."""

    total_paths: int
    paths_tested: int
    paths_passed: int
    paths_failed: int
    total_questions: int
    questions_passed: int
    questions_failed: int
    # SQL execution metrics
    sql_success_rate: float = 0.0  # Percentage of SQL queries that executed successfully
    sql_query_count: int = 0  # Total number of SQL queries executed
    # SQL data validation metrics
    sql_data_success_rate: float = 0.0  # Percentage of SQL results that passed validation
    sql_data_count: int = 0  # Number of questions with SQL data assertions
    # RAGAS metrics (0.0-1.0) - answer quality
    avg_relevance: float = 0.0  # RAGAS answer_relevancy
    avg_faithfulness: float = 0.0  # RAGAS faithfulness
    avg_answer_correctness: float = 0.0  # RAGAS answer_correctness
    # RAGAS metrics (0.0-1.0) - retrieval quality
    avg_account_precision: float = 0.0  # Account RAG precision
    avg_account_recall: float = 0.0  # Account RAG recall
    # Sample counts for N/A display (0 means no samples)
    account_sample_count: int = 0  # Number of steps with account chunks (RAG returned results)
    rag_invoked_count: int = 0  # Number of steps where RAG was actually invoked (includes empty results)
    # RAG detection accuracy (needs_rag decision accuracy)
    rag_detection_rate: float = 0.0  # Percentage of correct RAG decisions
    rag_detection_count: int = 0  # Number of questions with RAG expectations
    # RAGAS reliability tracking (per-metric, not per-call)
    ragas_metrics_total: int = 0  # Total individual metrics evaluated (questions × 5)
    ragas_metrics_failed: int = 0  # Individual metrics that returned NaN
    # Latency
    total_latency_ms: int = 0
    avg_latency_per_question_ms: float = 0.0
    p95_latency_ms: float = 0.0  # P95 latency per question
    wall_clock_ms: int = 0  # Total wall-clock time for the eval
    # Latency breakdown by node (percentage of total)
    latency_routing_pct: float = 0.0
    latency_retrieval_pct: float = 0.0
    latency_answer_pct: float = 0.0
    # Results
    failed_paths: list[FlowResult] = field(default_factory=list)
    all_results: list[FlowResult] = field(default_factory=list)

    @property
    def path_pass_rate(self) -> float:
        """Percentage of paths that passed."""
        return self.paths_passed / self.paths_tested if self.paths_tested > 0 else 0.0

    @property
    def question_pass_rate(self) -> float:
        """Percentage of questions that passed."""
        return self.questions_passed / self.total_questions if self.total_questions > 0 else 0.0

    @property
    def ragas_success_rate(self) -> float:
        """Percentage of RAGAS metrics that succeeded (1.0 = all succeeded, 0.0 = all failed)."""
        if self.ragas_metrics_total == 0:
            return 1.0  # No RAGAS metrics = no failures
        return (self.ragas_metrics_total - self.ragas_metrics_failed) / self.ragas_metrics_total

    @property
    def composite_score(self) -> float:
        """
        Weighted average of all quality metrics (0.0-1.0).

        Weights:
        - Faithfulness: 30% (critical - no hallucinations)
        - Answer Relevance: 20% (must address questions)
        - Answer Correctness: 15% (factual accuracy)
        - Account Precision: 10% (retrieved relevant chunks)
        - Account Recall: 10% (retrieved all needed chunks)
        - SQL Success: 10% (query generation)
        - Latency: 5% (speed, capped at SLO)
        """
        latency_score = _latency_score(self.avg_latency_per_question_ms, SLO_FLOW_AVG_LATENCY_MS)
        return (
            0.30 * self.avg_faithfulness
            + 0.20 * self.avg_relevance
            + 0.15 * self.avg_answer_correctness
            + 0.10 * self.avg_account_precision
            + 0.10 * self.avg_account_recall
            + 0.10 * self.sql_success_rate
            + 0.05 * latency_score
        )
