"""
LLM call helpers with retry logic and mock support.

This module provides functions for calling the LLM with
automatic retry on transient failures.

Uses LangChain LCEL chains and .with_structured_output() for reliable parsing.
"""

import logging
import os

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.agent.config import get_config, is_mock_mode
from backend.agent.schemas import Source

logger = logging.getLogger(__name__)


# =============================================================================
# Structured Output Models
# =============================================================================

class FollowUpSuggestions(BaseModel):
    """Structured output for follow-up question suggestions."""
    suggestions: list[str] = Field(
        description="List of 3 follow-up questions the user might want to ask next",
        min_length=1,
        max_length=5,
    )


# =============================================================================
# Follow-up Prompt Template
# =============================================================================

FOLLOW_UP_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful CRM assistant. Generate follow-up question suggestions based on the conversation context."),
    ("human", """Based on the user's question and conversation context, suggest 3 natural follow-up questions they might want to ask next.

User's current question: {question}
Mode used: {mode}
Company context: {company}

{conversation_history_section}

Generate 3 SHORT, SPECIFIC follow-up questions that would be valuable. Focus on:
- Drilling deeper into the data shown
- Related information they might need
- Actionable next steps
- Questions that build on the conversation context

IMPORTANT: If there's conversation history, suggest follow-ups that continue that flow naturally."""),
])


# Cached chains
_followup_chain = None
_answer_chain = None
_not_found_chain = None


def _get_followup_chain():
    """Get or create the cached follow-up chain with structured output."""
    global _followup_chain
    if _followup_chain is not None:
        return _followup_chain

    config = get_config()

    llm = ChatOpenAI(
        model=config.llm_model,
        temperature=0.7,  # Slightly creative for varied suggestions
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens=150,
    )

    # Use structured output for reliable list parsing
    structured_llm = llm.with_structured_output(FollowUpSuggestions)

    # Create LCEL chain: prompt | structured_llm
    _followup_chain = FOLLOW_UP_PROMPT_TEMPLATE | structured_llm

    logger.debug("Created follow-up chain with structured output")
    return _followup_chain


def _get_answer_chain():
    """Get or create the cached answer synthesis chain."""
    global _answer_chain
    if _answer_chain is not None:
        return _answer_chain

    from langchain_core.output_parsers import StrOutputParser
    from backend.agent.prompts import DATA_ANSWER_TEMPLATE

    config = get_config()

    llm = ChatOpenAI(
        model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create LCEL chain: prompt | llm | parser
    _answer_chain = DATA_ANSWER_TEMPLATE | llm | StrOutputParser()

    logger.debug("Created answer synthesis LCEL chain")
    return _answer_chain


def _get_not_found_chain():
    """Get or create the cached company not found chain."""
    global _not_found_chain
    if _not_found_chain is not None:
        return _not_found_chain

    from langchain_core.output_parsers import StrOutputParser
    from backend.agent.prompts import COMPANY_NOT_FOUND_TEMPLATE

    config = get_config()

    llm = ChatOpenAI(
        model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Create LCEL chain: prompt | llm | parser
    _not_found_chain = COMPANY_NOT_FOUND_TEMPLATE | llm | StrOutputParser()

    logger.debug("Created not-found LCEL chain")
    return _not_found_chain


def call_answer_chain(
    question: str,
    conversation_history_section: str,
    company_section: str,
    activities_section: str,
    history_section: str,
    pipeline_section: str,
    renewals_section: str,
    docs_section: str,
    account_context_section: str = "",
    contacts_section: str = "",
    groups_section: str = "",
    attachments_section: str = "",
) -> tuple[str, int]:
    """
    Call the answer synthesis chain using LCEL.

    Returns (answer_text, latency_ms).
    """
    import time

    if is_mock_mode():
        return mock_llm_response(question), 100

    chain = _get_answer_chain()

    start_time = time.time()
    answer = chain.invoke({
        "question": question,
        "conversation_history_section": conversation_history_section,
        "company_section": company_section,
        "contacts_section": contacts_section,
        "activities_section": activities_section,
        "history_section": history_section,
        "pipeline_section": pipeline_section,
        "renewals_section": renewals_section,
        "groups_section": groups_section,
        "attachments_section": attachments_section,
        "docs_section": docs_section,
        "account_context_section": account_context_section,
    })
    latency_ms = int((time.time() - start_time) * 1000)

    logger.info(f"Answer chain completed in {latency_ms}ms")
    return answer, latency_ms


def call_not_found_chain(
    question: str,
    query: str,
    matches: str,
) -> tuple[str, int]:
    """
    Call the company not found chain using LCEL.

    Returns (answer_text, latency_ms).
    """
    import time

    if is_mock_mode():
        return mock_llm_response("company not found"), 100

    chain = _get_not_found_chain()

    start_time = time.time()
    answer = chain.invoke({
        "question": question,
        "query": query,
        "matches": matches,
    })
    latency_ms = int((time.time() - start_time) * 1000)

    logger.info(f"Not-found chain completed in {latency_ms}ms")
    return answer, latency_ms


def mock_llm_response(prompt: str) -> str:
    """Generate a mock response for testing."""
    # Extract some context from the prompt for a semi-realistic response
    if "couldn't find an exact match" in prompt:
        return (
            "I couldn't find an exact match for that company in the CRM. "
            "Could you clarify which company you're asking about? "
            "Here are some similar companies I found that might be what you're looking for."
        )
    
    if "renewal" in prompt.lower():
        return (
            "**Upcoming Renewals Summary**\n\n"
            "Based on the CRM data, here are the accounts with upcoming renewals:\n\n"
            "• Several accounts have renewals coming up in the specified timeframe\n"
            "• Review each account's health status before the renewal date\n\n"
            "**Suggested Actions:**\n"
            "1. Schedule check-in calls with at-risk accounts\n"
            "2. Prepare renewal proposals for key accounts\n"
            "3. Review recent activity levels to identify any concerns"
        )
    
    if "pipeline" in prompt.lower():
        return (
            "**Pipeline Summary**\n\n"
            "Here's the current pipeline status based on CRM data:\n\n"
            "• Open opportunities are progressing through various stages\n"
            "• Total pipeline value and deal count are shown in the data\n\n"
            "**Suggested Actions:**\n"
            "1. Focus on deals in Proposal and Negotiation stages\n"
            "2. Follow up on stalled opportunities\n"
            "3. Update expected close dates if needed"
        )
    
    # Default response for company status questions
    return (
        "**Account Summary**\n\n"
        "Based on the CRM data provided:\n\n"
        "• Recent activities show engagement with the account\n"
        "• Pipeline includes open opportunities in various stages\n"
        "• History log shows recent touchpoints\n\n"
        "**Suggested Actions:**\n"
        "1. Review recent activity and follow up if needed\n"
        "2. Check opportunity progress and update stages\n"
        "3. Confirm next steps with key contacts"
    )


def call_docs_rag(question: str) -> tuple[str, list[Source]]:
    """
    Call the docs RAG pipeline from backend.rag.

    Returns (answer_text, doc_sources)
    """
    if is_mock_mode():
        return (
            "According to the documentation, you can find this feature "
            "in the Settings menu under Account Configuration.",
            [Source(type="doc", id="product_acme_crm_overview", label="Product Overview")]
        )

    try:
        from backend.rag.retrieval import create_backend
        from backend.rag.pipeline import answer_question as rag_answer

        config = get_config()
        backend = create_backend()
        # Use centralized config for RAG pipeline settings
        result = rag_answer(
            question,
            backend,
            use_hyde=config.rag_use_hyde,
            use_rewrite=config.rag_use_rewrite,
        )

        # Extract sources from used docs
        doc_sources = []
        for doc_id in result.get("doc_ids_used", [])[:3]:
            # Format doc_id into readable label
            label = doc_id.replace("_", " ").replace(".md", "").title()
            doc_sources.append(Source(type="doc", id=doc_id, label=label))

        return result.get("answer", ""), doc_sources

    except Exception as e:
        logger.warning(f"Docs RAG failed: {e}")
        return "", []


def call_account_rag(question: str, company_id: str) -> tuple[str, list[Source]]:
    """
    Call the account RAG pipeline for private CRM text search.

    Searches unstructured text (history notes, opportunity notes, attachments)
    scoped to a specific company.

    Args:
        question: The user's question
        company_id: Company ID to scope the search

    Returns:
        Tuple of (answer_text, account_sources)
    """
    if is_mock_mode():
        return (
            "Based on the account notes, the customer mentioned concerns about "
            "integration timeline during our last call.",
            [Source(type="account_note", id=f"{company_id}_notes", label="Account Notes")]
        )

    try:
        from backend.rag.pipeline.account import answer_account_question

        result = answer_account_question(
            question=question,
            company_id=company_id,
            include_docs=False,  # Only private data, docs handled separately
        )

        # Extract sources from account RAG
        account_sources = []
        for source in result.get("sources", [])[:5]:
            source_type = source.get("type", "account_note")
            source_id = source.get("id", "unknown")
            label = source.get("label", source_type.replace("_", " ").title())
            account_sources.append(Source(type=source_type, id=source_id, label=label))

        logger.info(
            f"Account RAG completed: {len(account_sources)} sources, "
            f"{result.get('meta', {}).get('latency_ms', 0)}ms"
        )

        return result.get("answer", ""), account_sources

    except Exception as e:
        logger.warning(f"Account RAG failed: {e}")
        return "", []


def generate_follow_up_suggestions(
    question: str,
    mode: str,
    company_id: str | None = None,
    conversation_history: str = "",
) -> list[str]:
    """
    Generate follow-up question suggestions using LLM with structured output.

    Uses LCEL chain with .with_structured_output() for reliable parsing.

    Args:
        question: The user's current question
        mode: The mode used (data, docs, data+docs)
        company_id: The company context if any
        conversation_history: Formatted conversation history for context

    Returns:
        List of 3 suggested follow-up questions.
    """
    config = get_config()

    if not config.enable_follow_up_suggestions:
        return []

    if is_mock_mode():
        # Return context-aware mock suggestions
        if "renewal" in question.lower():
            return [
                "Which renewals are at risk?",
                "What's the total renewal value this quarter?",
                "Show me accounts with no recent activity",
            ]
        elif "pipeline" in question.lower():
            return [
                "Which deals are stalled?",
                "What's the forecast for this quarter?",
                "Show me deals closing this month",
            ]
        elif company_id:
            # Company-specific follow-ups
            return [
                "What are their recent activities?",
                "Show me their open opportunities",
                "Who are the key contacts?",
            ]
        else:
            return [
                "What are the recent activities?",
                "Show me the open opportunities",
                "Any upcoming renewals?",
            ]

    try:
        # Get the LCEL chain with structured output
        chain = _get_followup_chain()

        # Format conversation history section
        history_section = ""
        if conversation_history:
            history_section = f"=== RECENT CONVERSATION ===\n{conversation_history}"

        # Invoke chain with structured output
        result: FollowUpSuggestions = chain.invoke({
            "question": question,
            "mode": mode,
            "company": company_id or "None specified",
            "conversation_history_section": history_section,
        })

        logger.debug(f"Generated {len(result.suggestions)} follow-up suggestions")
        return result.suggestions[:3]

    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
        return []


__all__ = [
    "call_docs_rag",
    "call_account_rag",
    "generate_follow_up_suggestions",
    # LCEL chain functions
    "call_answer_chain",
    "call_not_found_chain",
    # Structured output models
    "FollowUpSuggestions",
]
