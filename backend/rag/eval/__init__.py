# backend.rag.eval - Evaluation Utilities
"""
RAG evaluation utilities.

Modules:
- base: Console and common utilities
- parallel_runner: Parallel evaluation runner (used by agent e2e_eval)

Note: Standalone RAG evals (docs_eval, account_eval) have been removed.
All evaluation is now done through backend.agent.eval.e2e_eval and flow_eval.
"""

from backend.rag.eval.parallel_runner import run_parallel_evaluation

__all__ = [
    "run_parallel_evaluation",
]
