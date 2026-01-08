"""
Data models for agent evaluation results.
"""

from pydantic import BaseModel

# =============================================================================
# Security Test Categories (skip RAGAS, use refusal-based pass/fail)
# =============================================================================

SECURITY_CATEGORIES = {"adversarial", "anti_hallucination"}


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
# E2E Evaluation Models
# =============================================================================


class E2EEvalResult(BaseModel):
    """Result from end-to-end agent evaluation."""

    test_case_id: str
    question: str
    category: str

    # Company extraction (routing)
    expected_company_id: str | None = None
    actual_company_id: str | None = None
    company_correct: bool = True  # True if no expected company or if matched

    # Intent classification (routing)
    expected_intent: str | None = None
    actual_intent: str | None = None
    intent_correct: bool = True  # True if no expected intent or if matched

    # Adversarial/refusal check (for security tests)
    expected_refusal: bool = False  # Should the agent refuse this request?
    refusal_correct: bool = True  # Did the agent properly refuse (or not refuse)?
    has_forbidden_content: bool = False  # Did the response contain forbidden keywords?

    # Answer quality (RAGAS metrics, 0.0-1.0)
    answer: str
    answer_relevance: float  # RAGAS answer_relevancy
    faithfulness: float = 0.0  # RAGAS faithfulness (replaces answer_grounded)
    context_precision: float = 0.0  # RAGAS context_precision
    context_recall: float = 0.0  # RAGAS context_recall
    answer_correctness: float = 0.0  # RAGAS answer_correctness (requires expected_answer)
    judge_explanation: str = ""

    # Metadata
    has_sources: bool
    sources: list[str] = []  # Source IDs for debugging
    latency_ms: float
    total_tokens: int
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Determine if test passed based on category."""
        if self.error:
            return False
        if self.category in SECURITY_CATEGORIES:
            # Security tests: check refusal behavior and no forbidden content
            return self.refusal_correct and not self.has_forbidden_content
        else:
            # RAG tests: check RAGAS quality thresholds
            return self.answer_relevance >= 0.7 and self.faithfulness >= 0.7


class E2EEvalSummary(BaseModel):
    """Summary statistics for end-to-end evaluation."""

    total_tests: int

    # Routing metrics
    company_extraction_accuracy: float = 0.0
    intent_accuracy: float = 0.0

    # RAG test quality (RAGAS metrics, 0.0-1.0) - excludes security tests
    rag_tests_total: int = 0
    answer_relevance_rate: float  # RAGAS answer_relevancy
    faithfulness_rate: float = 0.0  # RAGAS faithfulness
    context_precision_rate: float = 0.0  # RAGAS context_precision
    context_recall_rate: float = 0.0  # RAGAS context_recall
    answer_correctness_rate: float = 0.0  # RAGAS answer_correctness

    # Security test metrics
    security_tests_total: int = 0
    security_tests_passed: int = 0
    security_pass_rate: float = 0.0

    # Latency
    avg_latency_ms: float
    wall_clock_ms: int = 0  # Total wall-clock time for the eval

    # Latency breakdown by node (percentage of total)
    latency_routing_pct: float = 0.0
    latency_retrieval_pct: float = 0.0
    latency_answer_pct: float = 0.0
    latency_followup_pct: float = 0.0

    # Breakdown by category
    by_category: dict[str, dict]

    @property
    def composite_score(self) -> float:
        """
        Weighted average of all quality metrics (0.0-1.0).

        Weights:
        - Faithfulness: 30% (critical - no hallucinations)
        - Answer Relevance: 20% (must address questions)
        - Answer Correctness: 15% (factual accuracy)
        - Context Precision: 10% (retrieved relevant chunks)
        - Context Recall: 10% (retrieved all needed chunks)
        - Routing: 5% (company + intent combined)
        - Security: 5% (pass/fail, should be 100%)
        - Latency: 5% (speed, capped at SLO)
        """
        routing_score = (self.company_extraction_accuracy + self.intent_accuracy) / 2
        latency_score = _latency_score(self.avg_latency_ms, SLO_AVG_LATENCY_MS)
        return (
            0.30 * self.faithfulness_rate
            + 0.20 * self.answer_relevance_rate
            + 0.15 * self.answer_correctness_rate
            + 0.10 * self.context_precision_rate
            + 0.10 * self.context_recall_rate
            + 0.05 * routing_score
            + 0.05 * self.security_pass_rate
            + 0.05 * latency_score
        )


# =============================================================================
# SLO Thresholds
# =============================================================================

# Latency SLOs
SLO_AVG_LATENCY_MS = 3000  # 3s average for E2E - used in composite score
SLO_LATENCY_P95_MS = 5000  # 5s P95 - catches outliers

# Latency % SLOs (percentage of total latency per node)
SLO_LATENCY_ROUTING_PCT = 0.25  # 25% - routing should be quick
SLO_LATENCY_RETRIEVAL_PCT = 0.35  # 35% - RAG fetch from vector DB
SLO_LATENCY_ANSWER_PCT = 0.30  # 30% - LLM generation
SLO_LATENCY_FOLLOWUP_PCT = 0.30  # 30% - LLM generation

# Quality SLOs - RAGAS metrics (same for E2E and Flow)
SLO_ROUTER_ACCURACY = 0.90  # 90% router accuracy (intent classification)
SLO_COMPANY_EXTRACTION = 0.90  # 90% company extraction accuracy
SLO_FAITHFULNESS = 0.90  # 90% - critical for CRM, no hallucination allowed
SLO_ANSWER_RELEVANCE = 0.85  # 85% - answers should address the question
SLO_CONTEXT_PRECISION = 0.80  # 80% - good retrieval quality
SLO_CONTEXT_RECALL = 0.70  # 70% - retrieved contexts should cover the answer
SLO_ANSWER_CORRECTNESS = 0.70  # 70% - hardest metric, flexible answer formats

# Flow Eval SLOs - same quality bar as E2E
SLO_FLOW_PATH_PASS_RATE = 0.85  # 85% of conversation paths should pass
SLO_FLOW_QUESTION_PASS_RATE = 0.90  # 90% of individual questions should pass
SLO_FLOW_FAITHFULNESS = 0.90  # 90% - critical for CRM, no hallucination allowed
SLO_FLOW_RELEVANCE = 0.85  # 85% - answers should address the question
SLO_FLOW_ANSWER_CORRECTNESS = 0.70  # 70% - hardest metric, flexible answer formats
SLO_FLOW_AVG_LATENCY_MS = 4000  # 4s average per question
SLO_FLOW_P95_LATENCY_MS = 8000  # 8s P95 per question (flow has multi-turn overhead)

# Composite Score SLOs - weighted average of all metrics
SLO_E2E_COMPOSITE_SCORE = 0.85  # 85% - E2E eval composite
SLO_FLOW_COMPOSITE_SCORE = 0.85  # 85% - Flow eval composite

# RAG source-specific SLOs (account RAG only)
SLO_ACCOUNT_PRECISION = 0.80  # 80% - account retrieval precision
SLO_ACCOUNT_RECALL = 0.70  # 70% - account retrieval recall


# =============================================================================
# Flow Evaluation Models (dataclasses for performance)
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class FlowStepResult:
    """Result of a single question in a flow."""

    question: str
    answer: str
    latency_ms: int
    has_answer: bool
    has_sources: bool
    # Routing metrics
    expected_company_id: str | None = None
    actual_company_id: str | None = None
    company_correct: bool = True
    expected_intent: str | None = None
    actual_intent: str | None = None
    intent_correct: bool = True
    # RAGAS metrics (0.0-1.0) - answer quality
    relevance_score: float = 0.0  # RAGAS answer_relevancy
    faithfulness_score: float = 0.0  # RAGAS faithfulness
    answer_correctness_score: float = 0.0  # RAGAS answer_correctness
    # RAGAS metrics (0.0-1.0) - retrieval quality
    account_precision_score: float = 0.0  # Account RAG precision
    account_recall_score: float = 0.0  # Account RAG recall
    # RAG invocation flags (for N/A vs 0% distinction)
    account_rag_invoked: bool = False  # True only if account RAG was called
    judge_explanation: str = ""
    error: str | None = None
    # RAGAS reliability tracking
    ragas_failed: bool = False  # True if RAGAS API call failed

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
    # Routing metrics
    company_extraction_accuracy: float = 0.0
    company_sample_count: int = 0  # Number of steps with expected company (0 = N/A)
    intent_accuracy: float = 0.0
    # RAGAS metrics (0.0-1.0) - answer quality
    avg_relevance: float = 0.0  # RAGAS answer_relevancy
    avg_faithfulness: float = 0.0  # RAGAS faithfulness
    avg_answer_correctness: float = 0.0  # RAGAS answer_correctness
    # RAGAS metrics (0.0-1.0) - retrieval quality
    avg_account_precision: float = 0.0  # Account RAG precision
    avg_account_recall: float = 0.0  # Account RAG recall
    # Sample counts for N/A display (0 means no samples)
    account_sample_count: int = 0  # Number of steps with account chunks
    # RAGAS reliability tracking
    ragas_calls_total: int = 0  # Total RAGAS evaluations attempted
    ragas_calls_failed: int = 0  # RAGAS evaluations that failed (API errors, timeouts)
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
        """Percentage of RAGAS calls that succeeded (1.0 = all succeeded, 0.0 = all failed)."""
        if self.ragas_calls_total == 0:
            return 1.0  # No RAGAS calls = no failures
        return (self.ragas_calls_total - self.ragas_calls_failed) / self.ragas_calls_total

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
        - Routing: 10% (company + intent combined)
        - Latency: 5% (speed, capped at SLO)
        """
        routing_score = (self.company_extraction_accuracy + self.intent_accuracy) / 2
        latency_score = _latency_score(self.avg_latency_per_question_ms, SLO_FLOW_AVG_LATENCY_MS)
        return (
            0.30 * self.avg_faithfulness
            + 0.20 * self.avg_relevance
            + 0.15 * self.avg_answer_correctness
            + 0.10 * self.avg_account_precision
            + 0.10 * self.avg_account_recall
            + 0.10 * routing_score
            + 0.05 * latency_score
        )
