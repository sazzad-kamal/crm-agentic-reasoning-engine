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

from backend.agent.config import get_config, is_mock_mode
from backend.agent.schemas import Source
from backend.agent.prompts import (
    FOLLOW_UP_PROMPT_TEMPLATE,
    DATA_ANSWER_TEMPLATE,
    COMPANY_NOT_FOUND_TEMPLATE,
)
from backend.agent.mocks import mock_llm_response


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
# Chain Factory
# =============================================================================


def _create_chain(
    prompt_template: Any,
    model_key: str = "main",
    structured_output: type[BaseModel] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    """
    Generic chain factory - eliminates duplicate chain creation.

    Args:
        prompt_template: LangChain prompt template
        model_key: "fast" for router model, "main" for primary model
        structured_output: Optional Pydantic model for structured output
        temperature: Override temperature (uses config default if None)
        max_tokens: Override max tokens (uses config default if None)

    Returns:
        LCEL chain: prompt | llm [| parser]
    """
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
    )

    if structured_output:
        return prompt_template | llm.with_structured_output(structured_output)

    return prompt_template | llm | StrOutputParser()


# =============================================================================
# Cached Chains (lazy initialization)
# =============================================================================


_chains_cache: dict[str, Any] = {}


def _get_chain(chain_type: str) -> Any:
    """Get or create a cached chain."""
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
    """Call the answer synthesis chain. Returns (answer_text, latency_ms)."""
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
    """Call the company not found chain. Returns (answer_text, latency_ms)."""
    if is_mock_mode():
        return mock_llm_response("company not found"), 100

    chain = _get_chain("not_found")
    start_time = time.time()

    answer = chain.invoke({"question": question, "query": query, "matches": matches})

    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Not-found chain completed in {latency_ms}ms")
    return answer, latency_ms


def call_docs_rag(question: str) -> tuple[str, list[Source]]:
    """Call the docs RAG tool. Returns (context_text, doc_sources)."""
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
    """Call the account RAG tool. Returns (context_text, account_sources)."""
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
    """
    Generate follow-up question suggestions.

    For demo reliability, uses hardcoded question tree by default.
    Falls back to LLM generation if question not in tree.
    """
    config = get_config()

    if not config.enable_follow_up_suggestions:
        return []

    # Try hardcoded tree first (100% reliable for demos)
    if use_hardcoded_tree:
        try:
            from backend.agent.question_tree import get_follow_ups, TERMINAL_FOLLOW_UPS
            follow_ups = get_follow_ups(question)
            if follow_ups and follow_ups != TERMINAL_FOLLOW_UPS:
                logger.debug(f"Using hardcoded follow-ups for: {question[:50]}...")
                return follow_ups[:3]
        except ImportError:
            logger.warning("Question tree not available, falling back to LLM")

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
    """Return context-aware mock suggestions: 2 grounded + 1 exploratory."""
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
    """Format available data counts into a readable string for the prompt."""
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
