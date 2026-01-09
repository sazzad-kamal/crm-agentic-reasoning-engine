"""
LLM-based router for intelligent question routing.

Uses an LLM to:
1. Understand query intent
2. Extract parameters (company, timeframe, etc.)
3. Expand/rewrite queries for better understanding

Uses LangChain's .with_structured_output() for reliable Pydantic parsing.
"""

import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.agent.core.config import get_config
from backend.agent.datastore import CRMDataStore, get_datastore
from backend.agent.route.prompts import ROUTER_EXAMPLES, ROUTER_PROMPT_TEMPLATE
from backend.agent.route.schemas import RouterResult

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Role-Based Starter Detection
# =============================================================================

# Maps starter question patterns to owner IDs
# Sales Rep = jsmith, CSM = amartin, Manager = None (sees all)
STARTER_OWNER_MAP = {
    # Sales Rep starters
    "how's my pipeline": "jsmith",
    "hows my pipeline": "jsmith",
    "how is my pipeline": "jsmith",
    "show my pipeline": "jsmith",
    "what's in my pipeline": "jsmith",
    # CSM starters
    "any renewals at risk": "amartin",
    "renewals at risk": "amartin",
    "which renewals are at risk": "amartin",
    "at-risk renewals": "amartin",
    "at risk renewals": "amartin",
    # Manager starters (no owner filter - sees all)
    "how's the team doing": None,
    "hows the team doing": None,
    "how is the team doing": None,
    "team performance": None,
    "how's my team": None,
}


def detect_owner_from_starter(question: str) -> str | None:
    q = question.lower().strip().rstrip("?")

    for pattern, owner in STARTER_OWNER_MAP.items():
        if pattern in q:
            logger.debug(f"Detected starter pattern '{pattern}' → owner={owner}")
            return owner

    return None


# =============================================================================
# LLM Router Implementation
# =============================================================================


class LLMRouterError(Exception):
    """Custom exception for LLM router failures."""

    pass


class LLMRouterResponse(BaseModel):
    """Pydantic model for structured LLM router output.

    Used with .with_structured_output() for reliable parsing.
    Includes intent classification and parameter extraction.
    """

    intent: Literal[
        # Company-specific (requires company name, triggers RAG)
        "company_overview",  # General company status, health, contacts
        "company_pipeline",  # Pipeline, deals for a specific company
        "company_activities",  # Recent activities for a company
        "company_history",  # Historical interactions, timeline
        # Aggregate/global (no specific company)
        "renewals",  # Contract renewals, expirations
        "pipeline_summary",  # Aggregate pipeline across all accounts
        "deals_at_risk",  # At-risk deals (stalled, overdue)
        "forecast",  # Pipeline forecast/projections
        "forecast_accuracy",  # Win rate and accuracy metrics
        "activities",  # Activity search (calls, emails, meetings)
        "contacts",  # Contact queries (lookup or search)
        "company_search",  # Search companies by criteria (segment, industry)
        "attachments",  # Document/file searches
        "analytics",  # Counts, breakdowns, aggregations
    ] = Field(default="pipeline_summary", description="The primary intent of the question")
    company_name: str | None = Field(
        default=None, description="The company/account name mentioned in the question, if any"
    )
    # Extracted parameters
    segment: str | None = Field(
        default=None, description="Company segment: Enterprise, Mid-Market, or SMB"
    )
    industry: str | None = Field(
        default=None,
        description="Industry: Software, Manufacturing, Healthcare, Food, Consulting, Retail",
    )
    role: str | None = Field(
        default=None, description="Contact role: Decision Maker, Champion, or Executive"
    )
    activity_type: str | None = Field(
        default=None, description="Activity type: Call, Email, Meeting, or Task"
    )
    analytics_metric: str | None = Field(
        default=None,
        description="Analytics metric: contact_breakdown, activity_breakdown, activity_count, accounts_by_group, pipeline_by_group",
    )
    analytics_group_by: str | None = Field(
        default=None, description="Group by field for analytics: role, type, or stage"
    )


# Cached structured LLM chain
_router_chain = None


def _get_router_chain():
    global _router_chain
    if _router_chain is not None:
        return _router_chain

    import os

    config = get_config()

    llm = ChatOpenAI(
        model=config.router_model,
        temperature=config.router_temperature,
        api_key=os.environ.get("OPENAI_API_KEY"),  # type: ignore[arg-type]
        max_retries=3,
    )

    # Use structured output for reliable Pydantic parsing
    structured_llm = llm.with_structured_output(LLMRouterResponse)

    # Create LCEL chain: prompt | structured_llm
    _router_chain = ROUTER_PROMPT_TEMPLATE | structured_llm

    logger.debug(f"Created router chain with model={config.router_model}")
    return _router_chain


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
def _call_llm_router(question: str, conversation_history: str = "") -> dict:
    logger.debug(f"LLM Router: Analyzing question: {question[:50]}...")

    # Build conversation context section
    conversation_context = ""
    if conversation_history:
        conversation_context = f"CONVERSATION HISTORY:\n{conversation_history}\n\n"

    # Get the cached chain
    chain = _get_router_chain()

    # Invoke the chain with structured output
    result: LLMRouterResponse = chain.invoke(
        {
            "examples": ROUTER_EXAMPLES,
            "conversation_context": conversation_context,
            "question": question,
        }
    )

    logger.debug(f"LLM Router: Structured output received: intent={result.intent}")

    return result.model_dump()


def llm_route_question(
    question: str,
    datastore: CRMDataStore | None = None,
    conversation_history: str = "",
) -> RouterResult:
    ds = datastore or get_datastore()

    # LLM routing (returns intent + company_name only)
    llm_result = _call_llm_router(question, conversation_history)

    logger.info(f"LLM Router: intent={llm_result['intent']}")

    # Resolve company ID if LLM found a company name
    resolved_company = None
    if llm_result.get("company_name"):
        resolved_company = ds.resolve_company_id(llm_result["company_name"])
        if resolved_company:
            logger.debug(
                f"Resolved company '{llm_result['company_name']}' to ID: {resolved_company}"
            )

    return RouterResult(
        company_id=resolved_company,
        intent=llm_result.get("intent", "pipeline_summary"),
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def route_question(
    question: str,
    datastore: CRMDataStore | None = None,
    conversation_history: str = "",
) -> RouterResult:
    """Route a question and return LLM-derived fields (company_id, intent)."""
    return llm_route_question(question, datastore, conversation_history)


__all__ = [
    "route_question",
    "llm_route_question",
    "detect_owner_from_starter",
    "LLMRouterError",
    "LLMRouterResponse",
]
