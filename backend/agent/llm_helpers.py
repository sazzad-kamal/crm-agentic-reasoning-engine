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
    ("system", "You are a helpful CRM assistant. Generate 3 follow-up question suggestions."),
    ("human", """Suggest 3 follow-up questions for the user.

User's question: {question}
Current company: {company}

=== AVAILABLE DATA FOR THIS COMPANY ===
{available_data}

{conversation_history_section}

GENERATE 3 QUESTIONS:
1. First question: Drill deeper into current company's available data (use company name)
2. Second question: Another angle on current company's data (use company name)
3. Third question: Let user explore something NEW - different company, general CRM question, or documentation topic

RULES:
- Questions 1-2: ONLY ask about data types listed as available above
- Question 3: Can be general (renewals, pipeline summary) or about CRM features
- Always use company name, not "they" or "their"
- Keep questions SHORT"""),
])


# Cached chains - set to None to force rebuild when module reloads
_followup_chain = None
_answer_chain = None
_not_found_chain = None


def clear_chain_caches():
    """Clear all cached chains. Call when prompts change."""
    global _followup_chain, _answer_chain, _not_found_chain
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
        model=config.router_model,  # Use fast model for suggestions (not critical path)
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
    company_name: str | None = None,
    conversation_history: str = "",
    available_data: dict | None = None,
    use_hardcoded_tree: bool = True,
) -> list[str]:
    """
    Generate follow-up question suggestions.

    For demo reliability, uses hardcoded question tree by default.
    Falls back to LLM generation if question not in tree.

    Args:
        question: The user's current question
        mode: The mode used (data, docs, data+docs)
        company_id: The company ID if any
        company_name: The company name for display
        conversation_history: Formatted conversation history for context
        available_data: Dict with counts of available data types
        use_hardcoded_tree: If True, use hardcoded tree (default for demos)

    Returns:
        List of 3 suggested follow-up questions.
    """
    config = get_config()

    if not config.enable_follow_up_suggestions:
        return []

    # Try hardcoded tree first (100% reliable for demos)
    if use_hardcoded_tree:
        try:
            from backend.agent.question_tree import get_follow_ups, TERMINAL_FOLLOW_UPS

            follow_ups = get_follow_ups(question)
            # If we got real follow-ups (not terminal), use them
            if follow_ups and follow_ups != TERMINAL_FOLLOW_UPS:
                logger.debug(f"Using hardcoded follow-ups for: {question[:50]}...")
                return follow_ups[:3]
            # For terminal questions, fall through to LLM/mock
        except ImportError:
            logger.warning("Question tree not available, falling back to LLM")

    # Format available data for the prompt
    data_context = _format_available_data(available_data, company_name)

    if is_mock_mode():
        # Return context-aware mock suggestions: 2 grounded + 1 exploratory
        suggestions = []
        name = company_name or "the account"

        # First 2: Grounded in available data
        if available_data:
            if available_data.get("opportunities", 0) > 0:
                suggestions.append(f"What stage are {name}'s opportunities in?")
            if available_data.get("activities", 0) > 0:
                suggestions.append(f"What were {name}'s recent activities?")
            if available_data.get("contacts", 0) > 0:
                suggestions.append(f"Who are {name}'s key contacts?")
            if available_data.get("renewals", 0) > 0:
                suggestions.append(f"When is {name}'s renewal coming up?")

        # Limit to 2 grounded questions
        suggestions = suggestions[:2]

        # Third: Always add an exploratory question
        suggestions.append("Show me the overall pipeline summary")

        return suggestions[:3]

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
            "company": company_name or company_id or "None specified",
            "available_data": data_context,
            "conversation_history_section": history_section,
        })

        logger.debug(f"Generated {len(result.suggestions)} follow-up suggestions")
        return result.suggestions[:3]

    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
        return []


def _format_available_data(data: dict | None, company_name: str | None) -> str:
    """Format available data counts into a readable string for the prompt."""
    if not data:
        return "No specific data available. Suggest general CRM questions."

    lines = []
    company_label = company_name or "this company"

    if data.get("contacts", 0) > 0:
        lines.append(f"- Contacts: {data['contacts']} contacts for {company_label}")
    if data.get("activities", 0) > 0:
        lines.append(f"- Activities: {data['activities']} recent activities")
    if data.get("opportunities", 0) > 0:
        lines.append(f"- Opportunities: {data['opportunities']} open opportunities")
    if data.get("history", 0) > 0:
        lines.append(f"- History: {data['history']} timeline entries")
    if data.get("renewals", 0) > 0:
        lines.append(f"- Renewals: {data['renewals']} upcoming renewals")
    if data.get("pipeline_summary"):
        lines.append("- Pipeline: Overall pipeline summary available")
    if data.get("docs", 0) > 0:
        lines.append(f"- Documentation: {data['docs']} relevant docs")

    if not lines:
        return "No specific data available. Suggest general CRM questions."

    return "\n".join(lines)


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
