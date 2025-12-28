"""
Constants for the pipeline module.
"""

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"  # Fast model for HyDE, query rewrite
ANSWER_MODEL = "gpt-4o"  # Best model for final answers
ANSWER_MAX_TOKENS = 800

# Context Building
MAX_CONTEXT_TOKENS = 3000
MAX_CHUNKS_PER_DOC = 3
MAX_CHUNKS_PER_TYPE = 4
MIN_BM25_SCORE_RATIO = 0.1

# =============================================================================
# Latency Budgets (milliseconds)
# =============================================================================
# Used for performance monitoring and alerting
LATENCY_BUDGETS: dict[str, int] = {
    "preprocess": 50,       # Query preprocessing
    "rewrite": 400,         # Query rewrite LLM call
    "hyde": 500,            # HyDE generation LLM call
    "retrieval": 300,       # Dense + BM25 retrieval
    "filter": 50,           # Gating and filtering
    "context": 50,          # Context building
    "generate": 3000,       # Answer generation LLM call
}

# Total pipeline budget (P95 SLO)
TOTAL_LATENCY_BUDGET_MS = 5000
