"""
Schema-driven query planner for LLM-based SQL generation.

Replaces the 14-intent router with direct SQL generation.
The LLM becomes a "CRM SQL expert" that understands the data model.
Loads prompt from prompt.txt for clean separation.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.agent.core.config import get_config
from backend.utils.prompt import load_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================


class SQLQuery(BaseModel):
    """A single SQL query with its purpose."""

    sql: str = Field(description="The SQL query to execute against DuckDB")
    purpose: str = Field(
        description="What this query fetches (e.g., 'company_info', 'open_deals', 'contact_info')"
    )


class QueryPlan(BaseModel):
    """LLM output containing SQL queries and RAG decision."""

    queries: list[SQLQuery] = Field(
        default_factory=list, description="List of SQL queries to execute"
    )
    needs_account_rag: bool = Field(
        default=False,
        description="True if question is about a specific company and needs unstructured context from RAG",
    )


# =============================================================================
# Role-Based Starter Detection (moved from router.py)
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
    """Detect owner ID from starter question patterns."""
    q = question.lower().strip().rstrip("?")

    for pattern, owner in STARTER_OWNER_MAP.items():
        if pattern in q:
            logger.debug(f"Detected starter pattern '{pattern}' → owner={owner}")
            return owner

    return None


# =============================================================================
# Schema Prompt Template (loaded from prompt.txt)
# =============================================================================

SCHEMA_PROMPT = load_prompt(Path(__file__).parent / "prompt.txt")


# =============================================================================
# LLM Chain Setup
# =============================================================================

_planner_chain = None


def _get_planner_chain():
    """Get or create the cached planner chain."""
    global _planner_chain
    if _planner_chain is not None:
        return _planner_chain

    config = get_config()

    llm = ChatOpenAI(
        model=config.router_model,
        temperature=0,  # Deterministic SQL generation
        api_key=os.environ.get("OPENAI_API_KEY"),  # type: ignore[arg-type]
        max_retries=3,
    )

    # Use structured output for reliable Pydantic parsing
    structured_llm = llm.with_structured_output(QueryPlan)

    # Create LCEL chain: prompt | structured_llm
    _planner_chain = SCHEMA_PROMPT | structured_llm

    logger.debug(f"Created query planner chain with model={config.router_model}")
    return _planner_chain


def reset_planner_chain() -> None:
    """Reset the cached planner chain (for testing)."""
    global _planner_chain
    _planner_chain = None


# =============================================================================
# Main Entry Point
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
def get_query_plan(
    question: str,
    conversation_history: str = "",
    owner: str | None = None,
) -> QueryPlan:
    """
    Get SQL query plan from LLM.

    Args:
        question: The user's question
        conversation_history: Formatted conversation history for context
        owner: Owner ID for filtering (e.g., "jsmith", "amartin")

    Returns:
        QueryPlan with SQL queries and needs_account_rag boolean
    """
    logger.debug(f"Query Planner: Analyzing question: {question[:50]}...")

    chain = _get_planner_chain()

    result: QueryPlan = chain.invoke(
        {
            "question": question,
            "conversation_history": conversation_history or "",
            "owner": owner or "all",
            "today": datetime.now().strftime("%Y-%m-%d"),
        }
    )

    logger.info(
        f"Query Planner: {len(result.queries)} queries, needs_rag={result.needs_account_rag}"
    )

    return result


__all__ = [
    "SQLQuery",
    "QueryPlan",
    "get_query_plan",
    "detect_owner_from_starter",
    "reset_planner_chain",
    "STARTER_OWNER_MAP",
]
