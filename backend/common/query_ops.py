"""
Shared query operations for RAG pipelines using LangChain.

Consolidates query rewriting and HyDE generation used by both
account and docs pipelines.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate

from backend.common.llm_client import call_llm_safe


logger = logging.getLogger(__name__)


# =============================================================================
# LangChain Prompt Templates
# =============================================================================

QUERY_REWRITE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant for a CRM documentation search system.
Your job is to take a user's question about Acme CRM Suite and rewrite it to be clearer and more specific.
Keep the rewritten query in natural language (not keywords).
If the query is already clear, return it mostly unchanged.
Only output the rewritten query, nothing else."""),
    ("human", "{prompt}"),
])

HYDE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are an expert on Acme CRM Suite documentation.
Given a question, write a short hypothetical answer (2-3 sentences) as if it came from the documentation.
This will be used for semantic search, so include relevant terminology and concepts.
Only output the hypothetical answer, nothing else."""),
    ("human", "{prompt}"),
])


# =============================================================================
# Legacy System Prompts (backwards compatibility)
# =============================================================================

QUERY_REWRITE_SYSTEM = """You are a query rewriting assistant for a CRM documentation search system.
Your job is to take a user's question about Acme CRM Suite and rewrite it to be clearer and more specific.
Keep the rewritten query in natural language (not keywords).
If the query is already clear, return it mostly unchanged.
Only output the rewritten query, nothing else."""

HYDE_SYSTEM = """You are an expert on Acme CRM Suite documentation.
Given a question, write a short hypothetical answer (2-3 sentences) as if it came from the documentation.
This will be used for semantic search, so include relevant terminology and concepts.
Only output the hypothetical answer, nothing else."""


# =============================================================================
# Query Operations
# =============================================================================

def rewrite_query(query: str, company_name: str | None = None) -> str:
    """
    Use LLM to rewrite vague queries into clearer ones.

    Args:
        query: Original user query
        company_name: Optional company context for account-specific queries

    Returns:
        Rewritten query (or original if rewriting fails)
    """
    context = f" for {company_name}" if company_name else ""
    logger.debug(f"Rewriting query{context}: {query[:50]}...")

    if company_name:
        prompt = f"Question about {company_name}: {query}"
    else:
        prompt = f"Rewrite this CRM question to be clearer: {query}"

    rewritten = call_llm_safe(
        prompt=prompt,
        system_prompt=QUERY_REWRITE_SYSTEM,
        max_tokens=150,
        default=query,
    )

    if rewritten != query:
        logger.debug(f"Query rewritten to: {rewritten[:50]}...")

    return rewritten


def generate_hyde(query: str, company_name: str | None = None) -> str:
    """
    Generate a hypothetical answer for HyDE retrieval.

    Args:
        query: The user's question
        company_name: Optional company context for account-specific queries

    Returns:
        A hypothetical answer to use for embedding
    """
    context = f" for {company_name}" if company_name else ""
    logger.debug(f"Generating HyDE{context}: {query[:50]}...")

    if company_name:
        prompt = f"Question about {company_name}: {query}"
    else:
        prompt = f"Question: {query}"

    hyde = call_llm_safe(
        prompt=prompt,
        system_prompt=HYDE_SYSTEM,
        max_tokens=200,
        default="",
    )

    if hyde:
        logger.debug(f"HyDE generated: {hyde[:50]}...")

    return hyde


__all__ = [
    # LangChain templates
    "QUERY_REWRITE_TEMPLATE",
    "HYDE_TEMPLATE",
    # Legacy prompts
    "QUERY_REWRITE_SYSTEM",
    "HYDE_SYSTEM",
    # Functions
    "rewrite_query",
    "generate_hyde",
]
