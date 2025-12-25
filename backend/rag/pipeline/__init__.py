# backend.rag.pipeline - RAG Pipelines
"""
RAG pipeline implementations.

Modules:
- base: Shared pipeline utilities (progress tracking, context building)
- docs: Documentation RAG pipeline (answer_question)
- account: Account-scoped RAG pipeline (answer_account_question)
- prompts: LLM prompt templates
"""

from backend.rag.pipeline.base import PipelineProgress, build_context
from backend.rag.pipeline.docs import answer_question
from backend.rag.pipeline.account import answer_account_question
from backend.rag.pipeline import prompts

__all__ = [
    "PipelineProgress",
    "build_context",
    "answer_question",
    "answer_account_question",
    "prompts",
]
