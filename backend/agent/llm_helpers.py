"""
LLM call helpers with retry logic and mock support.

This module provides functions for calling the LLM with
automatic retry on transient failures.
"""

import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from backend.agent.config import get_config, is_mock_mode
from backend.agent.schemas import Source

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_llm(prompt: str, system_prompt: str) -> tuple[str, int]:
    """
    Call the LLM with retry logic for transient failures.
    
    Returns (response_text, latency_ms)
    """
    config = get_config()
    
    if is_mock_mode():
        logger.debug("Mock mode: Returning mock LLM response")
        return mock_llm_response(prompt), 100
    
    # Import here to avoid loading OpenAI client when mocking
    from backend.common.llm_client import call_llm_with_metrics
    logger.debug(f"Calling LLM with model={config.llm_model}")
    
    result = call_llm_with_metrics(
        prompt=prompt,
        system_prompt=system_prompt,
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
        temperature=config.llm_temperature,
    )
    
    latency_ms = int(result["latency_ms"])
    logger.info(f"LLM response received in {latency_ms}ms")
    
    return result["response"], latency_ms


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
        
        backend = create_backend()
        # Enable HyDE and rewrite for better retrieval quality
        # Streaming mitigates the ~2-4 second latency impact
        result = rag_answer(question, backend, use_hyde=True, use_rewrite=True)
        
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


def generate_follow_up_suggestions(
    question: str,
    mode: str,
    company_id: str | None = None,
) -> list[str]:
    """
    Generate follow-up question suggestions using LLM.
    
    Returns list of 3 suggested follow-up questions.
    """
    from backend.agent.prompts import FOLLOW_UP_PROMPT
    
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
        else:
            return [
                "What are the recent activities?",
                "Show me the open opportunities",
                "Any upcoming renewals?",
            ]
    
    try:
        from backend.common.llm_client import call_llm as llm_call
        
        prompt = FOLLOW_UP_PROMPT.format(
            question=question,
            mode=mode,
            company=company_id or "None specified",
        )
        
        response = llm_call(
            prompt=prompt,
            system_prompt="You are a helpful CRM assistant.",
            model=config.llm_model,
            temperature=0.7,  # Slightly creative for varied suggestions
            max_tokens=150,
        )
        
        # Parse JSON array from response
        import json
        text = response.strip()
        if text.startswith("["):
            suggestions = json.loads(text)
            if isinstance(suggestions, list) and len(suggestions) >= 3:
                return suggestions[:3]
        
        logger.warning(f"Failed to parse follow-up suggestions: {text[:100]}")
        return []
        
    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
        return []


__all__ = [
    "call_llm",
    "call_docs_rag",
    "generate_follow_up_suggestions",
]
