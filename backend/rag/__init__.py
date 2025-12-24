"""RAG Pipeline for Acme CRM."""

from backend.rag.pipeline import answer_question, answer_account_question

__all__ = [
    "answer_question",
    "answer_account_question",
]
