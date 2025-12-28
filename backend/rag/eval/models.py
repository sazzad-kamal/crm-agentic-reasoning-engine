"""
Evaluation data models.

Defines the result structures for RAG evaluation.
"""

from pydantic import BaseModel


__all__ = [
    "JudgeResult",
    "EvalResult",
    "AccountEvalResult",
    "DocsEvalSummary",
    "AccountEvalSummary",
    "RAGEvalSummary",
    # Quality SLOs
    "SLO_CONTEXT_RELEVANCE",
    "SLO_ANSWER_RELEVANCE",
    "SLO_GROUNDEDNESS",
    "SLO_RAG_TRIAD",
    "SLO_DOC_RECALL",
    "SLO_PRIVACY_LEAKAGE",
    "SLO_NEGATIVE_HANDLING",
    # Production latency SLOs
    "SLO_LATENCY_P95_MS",
    "SLO_LATENCY_AVG_MS",
    # Eval latency SLOs
    "SLO_EVAL_LATENCY_P95_MS",
    "SLO_EVAL_LATENCY_AVG_MS",
]


# =============================================================================
# SLO Constants (Service Level Objectives)
# =============================================================================
# Based on industry standards for production RAG systems
# Reference: RAGAS benchmarks, enterprise AI quality standards

# Quality SLOs (computed on answerable questions only)
SLO_CONTEXT_RELEVANCE = 0.85    # 85% context relevance (retrieval quality)
SLO_ANSWER_RELEVANCE = 0.85     # 85% answer relevance (response quality)
SLO_GROUNDEDNESS = 0.85         # 85% groundedness (factual accuracy)
SLO_RAG_TRIAD = 0.80            # 80% full triad success (overall quality)
SLO_DOC_RECALL = 0.80           # 80% doc recall (retrieval coverage)
SLO_PRIVACY_LEAKAGE = 0.0       # 0% privacy leakage (strict - security)

# Negative question handling (correct rejection of unanswerable questions)
SLO_NEGATIVE_HANDLING = 0.90    # 90% correct rejection rate

# Production latency SLOs (what users experience)
SLO_LATENCY_P95_MS = 5000       # 5s P95 - catches outliers
SLO_LATENCY_AVG_MS = 3000       # 3s average - typical experience

# Eval latency SLOs (more lenient due to judge LLM overhead)
SLO_EVAL_LATENCY_P95_MS = 15000  # 15s P95 for eval (includes judge + RAGAS calls)
SLO_EVAL_LATENCY_AVG_MS = 10000  # 10s average for eval


class JudgeResult(BaseModel):
    """Result from LLM judge evaluation."""
    context_relevance: int  # 0 or 1
    answer_relevance: int   # 0 or 1
    groundedness: int       # 0 or 1
    needs_human_review: int # 0 or 1
    confidence: float = 0.5 # 0.0 to 1.0
    explanation: str = ""


class EvalResult(BaseModel):
    """Complete evaluation result for a single documentation question."""
    question_id: str
    question: str
    category: str = "single_doc"  # single_doc, multi_doc, negative, edge, etc.
    target_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    answer: str
    judge_result: JudgeResult
    doc_recall: float  # What fraction of target docs were retrieved
    latency_ms: float
    total_tokens: int
    step_timings: dict[str, float] = {}  # step_id -> elapsed_ms
    # Rerank scores for analysis
    max_rerank_score: float | None = None  # Highest rerank score from retrieval
    rerank_scores: list[float] = []  # All rerank scores (top-k)
    # RAGAS metrics (more accurate than binary judge scores)
    ragas_faithfulness: float | None = None  # 0-1, statement-level grounding
    ragas_answer_correctness: float | None = None  # 0-1, vs gold answer

    @property
    def is_negative(self) -> bool:
        """Check if this is a negative test (question about undocumented topic)."""
        return self.category == "negative" or len(self.target_doc_ids) == 0


class AccountEvalResult(BaseModel):
    """Evaluation result for a single account question."""
    question_id: str
    company_id: str
    company_name: str
    question: str
    question_type: str
    answer: str
    judge_result: JudgeResult
    privacy_leakage: int  # 1 if any retrieved chunk from wrong company
    leaked_company_ids: list[str]
    num_private_hits: int
    latency_ms: float
    total_tokens: int
    estimated_cost: float
    # RAGAS metrics
    ragas_faithfulness: float | None = None  # 0-1, statement-level grounding


# =============================================================================
# Summary Models
# =============================================================================

class DocsEvalSummary(BaseModel):
    """Summary of docs RAG evaluation."""
    total_tests: int
    answerable_tests: int = 0  # Tests with documented answers (excludes negative)
    negative_tests: int = 0    # Tests for undocumented topics
    context_relevance: float
    answer_relevance: float
    groundedness: float
    rag_triad_success: float
    avg_doc_recall: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    estimated_cost: float
    # RAGAS metrics (average across tests with scores)
    avg_ragas_faithfulness: float | None = None
    avg_ragas_answer_correctness: float | None = None
    # Negative question handling (correct rejection rate)
    negative_handling_rate: float | None = None  # % of negative Qs correctly declined
    # SLO tracking
    all_slos_passed: bool = True
    failed_slos: list[str] = []


class AccountEvalSummary(BaseModel):
    """Summary of account RAG evaluation."""
    total_tests: int
    context_relevance: float
    answer_relevance: float
    groundedness: float
    rag_triad_success: float
    privacy_leakage_rate: float
    leaked_questions: int
    avg_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    total_cost: float
    # SLO tracking
    all_slos_passed: bool = True
    failed_slos: list[str] = []


class RAGEvalSummary(BaseModel):
    """Combined RAG evaluation summary."""
    docs_eval: DocsEvalSummary | None = None
    account_eval: AccountEvalSummary | None = None
    overall_score: float = 0.0
    all_slos_passed: bool = True
    failed_slos: list[str] = []
    regression_detected: bool = False
    baseline_score: float | None = None
