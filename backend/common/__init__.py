# backend.common - Shared Utilities
"""
Shared utilities for the backend.

Modules:
- llm_client: OpenAI LLM client with retry logic
- company_resolver: Unified company name/ID resolution
- context_builder: RAG context building utilities
- prompts: Shared LLM prompt templates
- formatters: Data formatting for LLM prompts
- error_handling: Pipeline error handling decorators
"""

from backend.common.llm_client import call_llm, call_llm_safe, call_llm_with_metrics
from backend.common.company_resolver import (
    resolve_company_id,
    get_company_name,
    get_company_matches,
    CompanyResolver,
)
from backend.common.formatters import (
    format_company_section,
    format_activities_section,
    format_history_section,
    format_pipeline_section,
    format_renewals_section,
    format_docs_section,
)
from backend.common.error_handling import pipeline_node, safe_operation

# Context builder is imported lazily to avoid circular imports
# Import directly: from backend.common.context_builder import ContextBuilder

__all__ = [
    # LLM client
    "call_llm",
    "call_llm_safe",
    "call_llm_with_metrics",
    # Company resolver
    "resolve_company_id",
    "get_company_name",
    "get_company_matches",
    "CompanyResolver",
    # Formatters
    "format_company_section",
    "format_activities_section",
    "format_history_section",
    "format_pipeline_section",
    "format_renewals_section",
    "format_docs_section",
    # Error handling
    "pipeline_node",
    "safe_operation",
]
