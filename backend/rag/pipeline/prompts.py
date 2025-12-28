"""
Unified prompt templates for RAG answer generation using LangChain.

Query processing prompts (rewrite, HyDE) are in backend/common/query_ops.py.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# =============================================================================
# LangChain Prompt Templates
# =============================================================================

DOCS_ANSWER_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant answering questions about Acme CRM Suite.

CRITICAL INSTRUCTIONS:
1. ALWAYS answer using the provided context. Do not use outside knowledge.
2. If the context contains information related to the question - even partial or indirect - use it to construct an answer.
3. Cite your sources using [doc_id] format.
4. Be helpful and synthesize information from multiple sources when needed.

WHEN TO DECLINE:
- If the question asks about a topic, feature, or concept that is NOT mentioned anywhere in the context, say: "This topic is not documented in the available Acme CRM documentation."
- If the context only contains unrelated information (e.g., question asks about "mobile app" but context only discusses "reports"), you MUST decline.
- Do NOT make up information or guess. Only answer based on what's explicitly in the context."""),
    ("human", """Context from Acme CRM Suite documentation:
{context}

Question: {question}

Answer (with citations):"""),
])

ACCOUNT_ANSWER_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant answering questions about a specific customer account in a CRM system.

IMPORTANT RULES:
1. Use ONLY the provided context to answer. Do not use outside knowledge.
2. Look carefully for ANY relevant information in the context. Synthesize an answer from what's available, even if partial or indirect.
3. Only say "I don't see this in the available account data" if there is TRULY NO relevant information about the topic.
4. Cite your sources using [source_id] format.
5. Be concise but complete.
6. Focus on the specific account mentioned."""),
    ("human", """{context}

Question: {question}

Answer (with citations):"""),
])

HYBRID_ANSWER_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant answering questions about customer account '{company_name}' with CRM product documentation.

IMPORTANT RULES:
1. Use ONLY the provided context to answer. Do not use outside knowledge.
2. Look carefully for ANY relevant information in the context. Synthesize an answer from what's available, even if partial or indirect.
3. Only say "I don't see this in the available data" if there is TRULY NO relevant information about the topic.
4. Cite your sources using [source_id] format.
5. Be concise but complete.
6. Focus on the specific account when account data is available.
7. Reference documentation for product features and how-to guidance."""),
    ("human", """{context}

Question: {question}

Answer (with citations):"""),
])


# =============================================================================
# Backwards Compatible Functions
# =============================================================================

# Base answer prompt template (legacy string format)
_ANSWER_BASE = """You are an AI assistant answering questions about {domain}.

CRITICAL INSTRUCTIONS:
1. ALWAYS answer using the provided context. Do not use outside knowledge.
2. If the context contains information related to the question - even partial or indirect - use it to construct an answer.
3. Cite your sources using [{citation_format}] format.
4. Be helpful and synthesize information from multiple sources when needed.
{extra_rules}

WHEN TO DECLINE:
- If the question asks about a topic, feature, or concept NOT mentioned in the context, say: "This is not documented in the available information."
- If the context only contains unrelated information, you MUST decline rather than guess.
- Do NOT make up information. Only answer based on what's explicitly in the context.

{context}

Question: {question}

Answer (with citations):"""


def format_docs_answer_prompt(context: str, question: str) -> str:
    """Format prompt for documentation questions."""
    return _ANSWER_BASE.format(
        domain="Acme CRM Suite",
        citation_format="doc_id",
        extra_rules="",
        context=f"Context from Acme CRM Suite documentation:\n{context}",
        question=question,
    )


def format_account_answer_prompt(context: str, question: str) -> str:
    """Format prompt for account-scoped CRM questions."""
    return _ANSWER_BASE.format(
        domain="a specific customer account in a CRM system",
        citation_format="source_id",
        extra_rules="7. Focus on the specific account mentioned.",
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
        citation_format="source_id",
        extra_rules="""7. Focus on the specific account when account data is available.
8. Reference documentation for product features and how-to guidance.""",
        context=context,
        question=question,
    )


__all__ = [
    # LangChain templates
    "DOCS_ANSWER_TEMPLATE",
    "ACCOUNT_ANSWER_TEMPLATE",
    "HYBRID_ANSWER_TEMPLATE",
    # Backwards compatible functions
    "format_docs_answer_prompt",
    "format_account_answer_prompt",
    "format_hybrid_answer_prompt",
]
