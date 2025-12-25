"""
Prompt templates for RAG pipelines.

Centralizes all LLM prompts used across different pipeline modules.
"""


# =============================================================================
# Account Pipeline Prompts
# =============================================================================

QUERY_REWRITE_SYSTEM = """You are a query rewriting assistant for a CRM system.
Rewrite the user's question to be clearer and more specific for searching CRM records.
Keep it in natural language.
Only output the rewritten query, nothing else."""

HYDE_SYSTEM = """You are a CRM assistant. Given a question about a customer account,
write a short hypothetical answer (2-3 sentences) as if from CRM records.
Include relevant terms like history, notes, opportunities, activities.
Only output the hypothetical answer."""

ANSWER_SYSTEM = """You are an AI assistant answering questions about a specific customer account in a CRM system.

IMPORTANT RULES:
1. Use ONLY the provided context to answer. Do not use outside knowledge.
2. If the answer is not in the context, say "I don't see this in the available account data."
3. Cite your sources using [source_id] format, e.g., [history::HIST-ACME-CALL1] or [opp_note::OPP-ACME-UPGRADE].
4. Be concise but complete.
5. Focus on the specific account mentioned.

{context}

Question: {question}

Answer (with citations):"""


def format_account_answer_prompt(context: str, question: str) -> str:
    """Format the account answer prompt with context and question."""
    return ANSWER_SYSTEM.format(context=context, question=question)
