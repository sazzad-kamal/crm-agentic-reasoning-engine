"""
Prompt templates for RAG pipelines.

Centralizes all LLM prompts used across different pipeline modules.

NOTE: This module re-exports prompts from backend.common.prompts
for backwards compatibility. New code should import directly from there.
"""

# Re-export from common module
from backend.common.prompts import (
    QUERY_REWRITE_SYSTEM,
    HYDE_SYSTEM,
    HYDE_ACCOUNT_SYSTEM,
    format_account_answer_prompt,
    format_docs_answer_prompt,
    format_hybrid_answer_prompt,
    ANSWER_SYSTEM_DOCS,
    ANSWER_SYSTEM_ACCOUNT,
)

# Backwards compatibility alias
ANSWER_SYSTEM = ANSWER_SYSTEM_ACCOUNT
