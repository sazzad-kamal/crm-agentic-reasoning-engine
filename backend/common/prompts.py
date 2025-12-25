"""
Unified prompt templates and query helpers.

Consolidates prompts previously duplicated across:
- backend/rag/pipeline/docs.py
- backend/rag/pipeline/prompts.py
- backend/agent/prompts.py

Provides a single source of truth for all LLM prompts.
"""


# =============================================================================
# Query Processing Prompts
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


# Alternate version for CRM account-scoped queries
HYDE_ACCOUNT_SYSTEM = """You are a CRM assistant. Given a question about a customer account,
write a short hypothetical answer (2-3 sentences) as if from CRM records.
Include relevant terms like history, notes, opportunities, activities.
Only output the hypothetical answer."""


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


# =============================================================================
# Inline prompt strings (for backwards compatibility)
# =============================================================================

# Simple answer prompt without function formatting
ANSWER_SYSTEM_DOCS = """You are an AI assistant answering questions about Acme CRM Suite.

IMPORTANT RULES:
1. Use ONLY the provided context to answer. Do not use outside knowledge.
2. If the answer is not in the context, say "I don't see this documented in the provided sources."
3. Cite your sources using [doc_id] format, e.g., [opportunities_pipeline_and_forecasts].
4. Be concise but complete.
5. If multiple docs cover different aspects, synthesize the information and cite all relevant sources.

Context from Acme CRM Suite documentation:
{context}

Question: {question}

Answer (with citations):"""


ANSWER_SYSTEM_ACCOUNT = """You are an AI assistant answering questions about a specific customer account in a CRM system.

IMPORTANT RULES:
1. Use ONLY the provided context to answer. Do not use outside knowledge.
2. If the answer is not in the context, say "I don't see this in the available account data."
3. Cite your sources using [source_id] format, e.g., [history::HIST-ACME-CALL1] or [opp_note::OPP-ACME-UPGRADE].
4. Be concise but complete.
5. Focus on the specific account mentioned.

{context}

Question: {question}

Answer (with citations):"""
