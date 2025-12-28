"""
Unified prompt templates for answer generation.

Query processing prompts (rewrite, HyDE) are in backend/common/query_ops.py.
"""


# =============================================================================
# Answer Generation Prompts
# =============================================================================

# Base answer prompt template
_ANSWER_BASE = """You are an AI assistant answering questions about {domain}.

IMPORTANT RULES:
1. Use ONLY the provided context to answer. Do not use outside knowledge.
2. If the answer is not in the context, say "{not_found_message}"
3. Cite your sources using [{citation_format}] format.
4. Be concise but complete.
{extra_rules}

{context}

Question: {question}

Answer (with citations):"""


def format_docs_answer_prompt(context: str, question: str) -> str:
    """Format prompt for documentation questions."""
    return _ANSWER_BASE.format(
        domain="Acme CRM Suite",
        not_found_message="I don't see this documented in the provided sources.",
        citation_format="doc_id",
        extra_rules="5. If multiple docs cover different aspects, synthesize the information and cite all relevant sources.",
        context=f"Context from Acme CRM Suite documentation:\n{context}",
        question=question,
    )


def format_account_answer_prompt(context: str, question: str) -> str:
    """Format prompt for account-scoped CRM questions."""
    return _ANSWER_BASE.format(
        domain="a specific customer account in a CRM system",
        not_found_message="I don't see this in the available account data.",
        citation_format="source_id",
        extra_rules="5. Focus on the specific account mentioned.",
        context=context,
        question=question,
    )


def format_hybrid_answer_prompt(
    context: str,
    question: str,
    company_name: str = "the account",
) -> str:
    """Format prompt for hybrid (account + docs) questions."""
    return _ANSWER_BASE.format(
        domain=f"customer account '{company_name}' with CRM product documentation",
        not_found_message="I don't see this in the available account data or documentation.",
        citation_format="source_id",
        extra_rules="""5. Focus on the specific account when account data is available.
6. Reference documentation for product features and how-to guidance.""",
        context=context,
        question=question,
    )


__all__ = [
    "format_docs_answer_prompt",
    "format_account_answer_prompt",
    "format_hybrid_answer_prompt",
]
