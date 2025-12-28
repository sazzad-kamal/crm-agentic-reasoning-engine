"""
Data models for agent evaluation results.
"""

from pydantic import BaseModel


class E2EEvalResult(BaseModel):
    """Result from end-to-end agent evaluation."""

    test_case_id: str
    question: str
    category: str

    # Routing evaluation
    expected_mode: str
    actual_mode: str
    mode_correct: bool = False
    expected_company_id: str | None = None
    actual_company_id: str | None = None
    company_correct: bool = True  # True if no expected company or if matched

    # Tool selection
    expected_tools: list[str]
    actual_tools: list[str]
    tool_selection_correct: bool

    # Adversarial/refusal check (for security tests)
    expected_refusal: bool = False  # Should the agent refuse this request?
    refusal_correct: bool = True  # Did the agent properly refuse (or not refuse)?
    has_forbidden_content: bool = False  # Did the response contain forbidden keywords?

    # Answer quality (LLM-judged)
    answer: str
    answer_relevance: int  # 0 or 1
    answer_grounded: int  # 0 or 1
    judge_explanation: str = ""

    # Metadata
    has_sources: bool
    latency_ms: float
    total_tokens: int
    error: str | None = None


class E2EEvalSummary(BaseModel):
    """Summary statistics for end-to-end evaluation."""

    total_tests: int

    # Routing metrics
    mode_accuracy: float = 0.0
    company_extraction_accuracy: float = 0.0

    # Answer quality
    answer_relevance_rate: float
    groundedness_rate: float
    tool_selection_accuracy: float

    # Latency
    avg_latency_ms: float
    p95_latency_ms: float = 0.0
    latency_slo_pass: bool = True

    # Breakdown by category
    by_category: dict[str, dict]
    by_mode: dict[str, dict] = {}  # mode -> {expected, correct, accuracy}


# SLO Thresholds
SLO_LATENCY_P95_MS = 5000  # 5 second p95 latency
SLO_MODE_ACCURACY = 0.90  # 90% routing accuracy
SLO_ANSWER_RELEVANCE = 0.80  # 80% answer relevance
SLO_GROUNDEDNESS = 0.80  # 80% groundedness
SLO_OVERALL = 0.80  # 80% overall


class AgentEvalSummary(BaseModel):
    """Complete agent evaluation summary."""

    e2e_eval: E2EEvalSummary | None = None
    overall_score: float = 0.0  # Weighted composite score
    all_slos_passed: bool = True  # Did all SLOs pass?
    failed_slos: list[str] = []  # Which SLOs failed?
    regression_detected: bool = False  # Score worse than baseline?
    baseline_score: float | None = None  # Previous run score
