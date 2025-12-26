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
    "SLO_CONTEXT_RELEVANCE",
    "SLO_ANSWER_RELEVANCE",
    "SLO_GROUNDEDNESS",
    "SLO_RAG_TRIAD",
    "SLO_DOC_RECALL",
    "SLO_PRIVACY_LEAKAGE",
    "SLO_LATENCY_P95_MS",
]


# =============================================================================
# SLO Constants (Service Level Objectives)
# =============================================================================
SLO_CONTEXT_RELEVANCE = 0.80    # 80% context relevance
SLO_ANSWER_RELEVANCE = 0.80     # 80% answer relevance
SLO_GROUNDEDNESS = 0.80         # 80% groundedness
SLO_RAG_TRIAD = 0.70            # 70% full triad success
SLO_DOC_RECALL = 0.70           # 70% doc recall
SLO_PRIVACY_LEAKAGE = 0.0       # 0% privacy leakage (strict)
SLO_LATENCY_P95_MS = 5000       # 5 second P95 latency


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
    target_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    answer: str
    judge_result: JudgeResult
    doc_recall: float  # What fraction of target docs were retrieved
    latency_ms: float
    total_tokens: int


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


# =============================================================================
# Summary Models
# =============================================================================

class DocsEvalSummary(BaseModel):
    """Summary of docs RAG evaluation."""
    total_tests: int
    context_relevance: float
    answer_relevance: float
    groundedness: float
    rag_triad_success: float
    avg_doc_recall: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    estimated_cost: float
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
