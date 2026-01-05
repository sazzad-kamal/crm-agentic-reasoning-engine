"""
LLM call helpers with chain factory pattern.

Uses a factory pattern to eliminate duplicate chain creation code
while providing retry logic and mock support.
"""

import logging
import os
import time
from typing import Callable, Any

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from backend.agent.core.config import get_config, is_mock_mode
from backend.agent.core.schemas import Source
from backend.agent.llm.prompts import (
    FOLLOW_UP_PROMPT_TEMPLATE,
    DATA_ANSWER_TEMPLATE,
    COMPANY_NOT_FOUND_TEMPLATE,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Mock Responses (for testing without LLM API calls)
# =============================================================================


def mock_llm_response(prompt: str) -> str:
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
# Chain Factory
# =============================================================================


def _create_chain(
    prompt_template: Any,
    model_key: str = "main",
    structured_output: type[BaseModel] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    config = get_config()

    # Select model based on key
    model = config.router_model if model_key == "fast" else config.llm_model

    # Use overrides or config defaults
    temp = temperature if temperature is not None else config.llm_temperature
    tokens = max_tokens if max_tokens is not None else config.llm_max_tokens

    llm = ChatOpenAI(
        model=model,
        temperature=temp,
        max_tokens=tokens,
        api_key=os.environ.get("OPENAI_API_KEY"),
        streaming=True,  # Enable token-by-token streaming
    )

    if structured_output:
        return prompt_template | llm.with_structured_output(structured_output)

    return prompt_template | llm | StrOutputParser()


# =============================================================================
# Cached Chains (lazy initialization)
# =============================================================================


_chains_cache: dict[str, Any] = {}


def _get_chain(chain_type: str) -> Any:
    if chain_type in _chains_cache:
        return _chains_cache[chain_type]

    # Chain definitions
    chain_defs: dict[str, Callable[[], Any]] = {
        "followup": lambda: _create_chain(
            FOLLOW_UP_PROMPT_TEMPLATE,
            model_key="fast",
            structured_output=FollowUpSuggestions,
            temperature=0.7,
            max_tokens=150,
        ),
        "answer": lambda: _create_chain(
            DATA_ANSWER_TEMPLATE,
            model_key="main",
        ),
        "not_found": lambda: _create_chain(
            COMPANY_NOT_FOUND_TEMPLATE,
            model_key="main",
        ),
    }

    if chain_type not in chain_defs:
        raise ValueError(f"Unknown chain type: {chain_type}")

    chain = chain_defs[chain_type]()
    _chains_cache[chain_type] = chain
    logger.debug(f"Created {chain_type} chain")
    return chain


# =============================================================================
# Public API
# =============================================================================


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
    if is_mock_mode():
        return mock_llm_response(question), 100

    chain = _get_chain("answer")
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


def call_not_found_chain(question: str, query: str, matches: str) -> tuple[str, int]:
    if is_mock_mode():
        return mock_llm_response("company not found"), 100

    chain = _get_chain("not_found")
    start_time = time.time()

    answer = chain.invoke({"question": question, "query": query, "matches": matches})

    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Not-found chain completed in {latency_ms}ms")
    return answer, latency_ms


def call_docs_rag(question: str) -> tuple[str, list[Source]]:
    if is_mock_mode():
        return (
            "According to the documentation, you can find this feature "
            "in the Settings menu under Account Configuration.",
            [Source(type="doc", id="product_acme_crm_overview", label="Product Overview")],
        )

    try:
        from backend.agent.rag.tools import tool_docs_rag
        return tool_docs_rag(question)
    except Exception as e:
        logger.warning(f"Docs RAG failed: {e}")
        return "", []


def call_account_rag(question: str, company_id: str) -> tuple[str, list[Source]]:
    if is_mock_mode():
        return (
            "Based on the account notes, the customer mentioned concerns about "
            "integration timeline during our last call.",
            [Source(type="account_note", id=f"{company_id}_notes", label="Account Notes")],
        )

    try:
        from backend.agent.rag.tools import tool_account_rag
        return tool_account_rag(question, company_id)
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
    config = get_config()

    if not config.enable_follow_up_suggestions:
        return []

    # Try hardcoded tree first (100% reliable for demos)
    if use_hardcoded_tree:
        from backend.agent.question_tree import get_follow_ups
        follow_ups = get_follow_ups(question)
        if follow_ups:  # Empty list means terminal/leaf node
            logger.debug(f"Using hardcoded follow-ups for: {question[:50]}...")
            return follow_ups[:3]

    # Mock mode fallback
    if is_mock_mode():
        return _get_mock_suggestions(company_name, available_data)

    # LLM generation
    try:
        chain = _get_chain("followup")
        data_context = _format_available_data(available_data, company_name)
        history_section = f"=== RECENT CONVERSATION ===\n{conversation_history}" if conversation_history else ""

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


def _get_mock_suggestions(company_name: str | None, available_data: dict | None) -> list[str]:
    suggestions = []
    name = company_name or "the account"

    if available_data:
        if available_data.get("opportunities", 0) > 0:
            suggestions.append(f"What stage are {name}'s opportunities in?")
        if available_data.get("activities", 0) > 0:
            suggestions.append(f"What were {name}'s recent activities?")
        if available_data.get("contacts", 0) > 0:
            suggestions.append(f"Who are {name}'s key contacts?")
        if available_data.get("renewals", 0) > 0:
            suggestions.append(f"When is {name}'s renewal coming up?")

    suggestions = suggestions[:2]
    suggestions.append("Show me the overall pipeline summary")
    return suggestions[:3]


def _format_available_data(data: dict | None, company_name: str | None) -> str:
    if not data:
        return "No specific data available. Suggest general CRM questions."

    lines = []
    company_label = company_name or "this company"

    mappings = [
        ("contacts", f"Contacts: {{}} contacts for {company_label}"),
        ("activities", "Activities: {} recent activities"),
        ("opportunities", "Opportunities: {} open opportunities"),
        ("history", "History: {} timeline entries"),
        ("renewals", "Renewals: {} upcoming renewals"),
    ]

    for key, template in mappings:
        if data.get(key, 0) > 0:
            lines.append(f"- {template.format(data[key])}")

    if data.get("pipeline_summary"):
        lines.append("- Pipeline: Overall pipeline summary available")
    if data.get("docs", 0) > 0:
        lines.append(f"- Documentation: {data['docs']} relevant docs")

    return "\n".join(lines) if lines else "No specific data available. Suggest general CRM questions."


__all__ = [
    "call_docs_rag",
    "call_account_rag",
    "generate_follow_up_suggestions",
    "call_answer_chain",
    "call_not_found_chain",
    "FollowUpSuggestions",
]
