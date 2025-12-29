"""
Data models for agent evaluation results.
"""

from pydantic import BaseModel


# =============================================================================
# Tool Evaluation Models
# =============================================================================

class ToolEvalResult(BaseModel):
    """Result from individual tool evaluation."""

    tool_name: str
    test_case_id: str
    input_params: dict = {}

    # Expected vs actual
    expected_found: bool = True
    actual_found: bool = False
    expected_company_id: str | None = None
    actual_company_id: str | None = None

    # Quality metrics
    data_correct: bool = False
    sources_present: bool = False
    latency_ms: float = 0.0
    error: str | None = None


class ToolEvalSummary(BaseModel):
    """Summary statistics for tool evaluation."""

    total_tests: int
    passed: int = 0
    failed: int = 0
    accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    by_tool: dict[str, dict] = {}


# =============================================================================
# Router Evaluation Models
# =============================================================================

class RouterEvalResult(BaseModel):
    """Result from router evaluation."""

    test_case_id: str
    question: str
    expected_mode: str
    actual_mode: str = ""
    expected_company_id: str | None = None
    actual_company_id: str | None = None
    mode_correct: bool = False
    company_correct: bool = True
    intent_expected: str | None = None
    intent_actual: str | None = None
    intent_correct: bool = True
    latency_ms: float = 0.0
    error: str | None = None


class RouterEvalSummary(BaseModel):
    """Summary statistics for router evaluation."""

    total_tests: int
    mode_accuracy: float = 0.0
    company_extraction_accuracy: float = 0.0
    intent_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    by_mode: dict[str, dict] = {}


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

    # Answer quality (LLM-judged) - RAGAS-style metrics
    answer: str
    answer_relevance: int  # 0 or 1 - Does answer address the question?
    answer_grounded: int  # 0 or 1 - Is answer grounded in expected data type?
    context_relevance: int = 0  # 0 or 1 - Was retrieved context relevant to question?
    faithfulness: int = 0  # 0 or 1 - Is answer faithful to retrieved context (no hallucination)?
    judge_explanation: str = ""

    # Metadata
    has_sources: bool
    sources: list[str] = []  # Source IDs for debugging
    latency_ms: float
    total_tokens: int
    error: str | None = None


class E2EEvalSummary(BaseModel):
    """Summary statistics for end-to-end evaluation."""

    total_tests: int

    # Routing metrics
    company_extraction_accuracy: float = 0.0
    intent_accuracy: float = 0.0

    # Answer quality (RAGAS-style metrics)
    answer_relevance_rate: float  # Does answer address the question?
    groundedness_rate: float  # Is answer grounded in expected data type?
    context_relevance_rate: float = 0.0  # Was retrieved context relevant?
    faithfulness_rate: float = 0.0  # Is answer faithful to context (no hallucination)?

    # Latency
    avg_latency_ms: float
    p95_latency_ms: float = 0.0
    latency_slo_pass: bool = True

    # Breakdown by category
    by_category: dict[str, dict]


# =============================================================================
# SLO Thresholds
# =============================================================================

# Production latency SLOs (what users experience)
SLO_LATENCY_P95_MS = 5000       # 5s P95 - catches outliers
SLO_LATENCY_AVG_MS = 3000       # 3s average - typical experience

# Eval latency SLOs (more lenient due to judge LLM overhead)
SLO_EVAL_LATENCY_P95_MS = 10000  # 10s P95 for eval (includes ~500ms judge call)
SLO_EVAL_LATENCY_AVG_MS = 6000   # 6s average for eval

# Quality SLOs
SLO_TOOL_ACCURACY = 0.90        # 90% tool accuracy
SLO_ROUTER_ACCURACY = 0.90      # 90% router accuracy (company extraction)
SLO_ANSWER_RELEVANCE = 0.80     # 80% answer relevance
SLO_GROUNDEDNESS = 0.80         # 80% groundedness
SLO_OVERALL = 0.80              # 80% overall


class AgentEvalSummary(BaseModel):
    """Complete agent evaluation summary."""

    e2e_eval: E2EEvalSummary | None = None
    overall_score: float = 0.0  # Weighted composite score
    all_slos_passed: bool = True  # Did all SLOs pass?
    failed_slos: list[str] = []  # Which SLOs failed?
    regression_detected: bool = False  # Score worse than baseline?
    baseline_score: float | None = None  # Previous run score
