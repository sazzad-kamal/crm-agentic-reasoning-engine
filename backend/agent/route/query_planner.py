"""
Schema-driven query planner for LLM-based SQL generation.

Replaces the 14-intent router with direct SQL generation.
The LLM becomes a "CRM SQL expert" that understands the data model.
Loads prompt from prompt.txt for clean separation.

Supports both Chat Completions API and Responses API for different models.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.agent.core.config import get_config
from backend.agent.route.slot_query import SlotPlan, SlotQuery, slot_to_sql
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
    """LLM output containing SQL queries."""

    queries: list[SQLQuery] = Field(
        default_factory=list, description="List of SQL queries to execute"
    )


# =============================================================================
# Role-Based Starter Detection (moved from router.py)
# =============================================================================

# Maps starter question patterns to owner IDs
# Sales Rep = jsmith, CSM = amartin, Manager = None (sees all)
STARTER_OWNER_MAP = {
    # Sales Rep starters
    "show my open deals": "jsmith",
    "my open deals": "jsmith",
    "show my deals": "jsmith",
    # CSM starters
    "show my at-risk renewals": "amartin",
    "at-risk renewals": "amartin",
    "my at-risk renewals": "amartin",
    # Manager starters (no owner filter - sees all)
    "show team pipeline": None,
    "team pipeline": None,
    "show all deals": None,
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

SCHEMA_PROMPT_TEMPLATE = load_prompt(Path(__file__).parent / "prompt.txt")
SLOT_PROMPT_TEMPLATE = load_prompt(Path(__file__).parent / "slot_prompt.txt")

# Models that require Responses API instead of Chat Completions
RESPONSES_API_MODELS = frozenset({
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex",
    "gpt-5-codex-mini",
    "gpt-5-codex",
    "codex-mini-latest",
    "codex-1",
})


# =============================================================================
# OpenAI Client
# =============================================================================

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    """Get or create the cached OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def reset_planner_chain() -> None:
    """Reset the cached OpenAI client (for testing)."""
    global _openai_client
    _openai_client = None


# =============================================================================
# API Callers
# =============================================================================


def _call_chat_completions(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> QueryPlan:
    """Call Chat Completions API with structured output."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "QueryPlan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sql": {"type": "string"},
                                    "purpose": {"type": "string"},
                                },
                                "required": ["sql", "purpose"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["queries"],
                    "additionalProperties": False,
                },
            },
        },
    )

    content = response.choices[0].message.content
    if content:
        data = json.loads(content)
        return QueryPlan(**data)
    return QueryPlan(queries=[])


def _call_responses_api(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> QueryPlan:
    """Call Responses API with structured output for Codex models."""
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "QueryPlan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sql": {"type": "string"},
                                    "purpose": {"type": "string"},
                                },
                                "required": ["sql", "purpose"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["queries"],
                    "additionalProperties": False,
                },
            },
        },
    )

    # Extract text from response
    content = response.output_text
    if content:
        data = json.loads(content)
        return QueryPlan(**data)
    return QueryPlan(queries=[])


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
    error_feedback: str | None = None,
) -> QueryPlan:
    """
    Get SQL query plan from LLM.

    Args:
        question: The user's question
        conversation_history: Formatted conversation history for context
        owner: Owner ID for filtering (e.g., "jsmith", "amartin")
        error_feedback: Optional error message from failed SQL execution for retry

    Returns:
        QueryPlan with SQL queries
    """
    logger.debug(f"Query Planner: Analyzing question: {question[:50]}...")

    config = get_config()
    client = _get_openai_client()
    model = config.router_model

    # Build conversation history with error feedback if present
    history = conversation_history or ""
    if error_feedback:
        history = f"{history}\n\n[PREVIOUS SQL FAILED]\n{error_feedback}\nPlease fix the SQL query."
        logger.info(f"Query Planner: Retrying with error feedback: {error_feedback[:100]}...")

    # Format the prompt
    system_prompt = SCHEMA_PROMPT_TEMPLATE.format(
        today=datetime.now().strftime("%Y-%m-%d"),
        owner=owner or "all",
        conversation_history=history,
        question=question,
    )

    # Use appropriate API based on model
    if model in RESPONSES_API_MODELS:
        logger.debug(f"Using Responses API for model={model}")
        result = _call_responses_api(client, model, system_prompt, question)
    else:
        logger.debug(f"Using Chat Completions API for model={model}")
        result = _call_chat_completions(client, model, system_prompt, question)

    logger.info(f"Query Planner: {len(result.queries)} queries")

    # Log each SQL query for troubleshooting (debug level - only shows with -v)
    for i, q in enumerate(result.queries, 1):
        logger.debug(f"SQL [{i}] ({q.purpose}): {q.sql}")

    return result


# =============================================================================
# Slot-Based Query Planning
# =============================================================================

# JSON schema for SlotPlan structured output
# All filter keys must be explicitly defined for strict mode
SLOT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "enum": [
                            "opportunities",
                            "contacts",
                            "activities",
                            "companies",
                            "history",
                            "attachments",
                        ],
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            # Opportunities filters
                            "owner": {"type": ["string", "null"]},
                            "stage": {"type": ["string", "null"]},
                            "stage_not_in": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                            "type": {"type": ["string", "null"]},
                            "value_gt": {"type": ["number", "null"]},
                            "value_lt": {"type": ["number", "null"]},
                            # Contacts/Activities/Companies shared
                            "company_name": {"type": ["string", "null"]},
                            "company_id": {"type": ["string", "null"]},
                            # Contacts filters
                            "role": {"type": ["string", "null"]},
                            "lifecycle_stage": {"type": ["string", "null"]},
                            # Activities filters
                            "contact_id": {"type": ["string", "null"]},
                            "date_after": {"type": ["string", "null"]},
                            # Companies filters
                            "name": {"type": ["string", "null"]},
                            "status": {"type": ["string", "null"]},
                            "health_flags": {"type": ["string", "null"]},
                            "account_owner": {"type": ["string", "null"]},
                        },
                        "required": [
                            "owner", "stage", "stage_not_in", "type", "value_gt", "value_lt",
                            "company_name", "company_id", "role", "lifecycle_stage",
                            "contact_id", "date_after", "name", "status", "health_flags", "account_owner"
                        ],
                        "additionalProperties": False,
                    },
                    "columns": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "order_by": {
                        "type": ["string", "null"],
                    },
                    "purpose": {
                        "type": "string",
                    },
                },
                "required": ["table", "filters", "columns", "order_by", "purpose"],
                "additionalProperties": False,
            },
        },
    },
    "needs_rag": {
        "type": "boolean",
        "description": "True if question needs RAG context (why, what happened, concerns). False for list/show/count questions.",
    },
    "required": ["queries", "needs_rag"],
    "additionalProperties": False,
}


def _call_chat_completions_slots(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> SlotPlan:
    """Call Chat Completions API for slot-based output."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SlotPlan",
                "strict": True,
                "schema": SLOT_PLAN_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content
    if content:
        data = json.loads(content)
        queries = [SlotQuery(**q) for q in data.get("queries", [])]
        needs_rag = data.get("needs_rag", False)
        return SlotPlan(queries=queries, needs_rag=needs_rag)
    return SlotPlan(queries=[], needs_rag=False)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)
def get_slot_plan(
    question: str,
    conversation_history: str = "",
    owner: str | None = None,
) -> SlotPlan:
    """
    Get slot-based query plan from LLM.

    This is more reliable than raw SQL generation because:
    - LLM outputs structured slots (table, filters, order_by)
    - We build SQL programmatically - no syntax errors possible

    Args:
        question: The user's question
        conversation_history: Formatted conversation history for context
        owner: Owner ID for filtering (e.g., "jsmith", "amartin")

    Returns:
        SlotPlan with slot queries
    """
    logger.debug(f"Slot Planner: Analyzing question: {question[:50]}...")

    config = get_config()
    client = _get_openai_client()
    model = config.router_model

    # Format the prompt
    system_prompt = SLOT_PROMPT_TEMPLATE.format(
        today=datetime.now().strftime("%Y-%m-%d"),
        owner=owner or "all",
        conversation_history=conversation_history or "",
        question=question,
    )

    # Call LLM for slots
    result = _call_chat_completions_slots(client, model, system_prompt, question)

    logger.info(f"Slot Planner: {len(result.queries)} queries")

    # Log each slot query
    for i, q in enumerate(result.queries, 1):
        logger.debug(f"Slot [{i}] ({q.purpose}): table={q.table}, filters={q.filters}")

    return result


def slot_plan_to_query_plan(slot_plan: SlotPlan) -> QueryPlan:
    """
    Convert a SlotPlan to a QueryPlan by building SQL from slots.

    This allows slot-based planning to work with existing executor code.
    """
    queries = []
    for slot in slot_plan.queries:
        sql = slot_to_sql(slot)
        queries.append(SQLQuery(sql=sql, purpose=slot.purpose))
    return QueryPlan(queries=queries)


__all__ = [
    "SQLQuery",
    "QueryPlan",
    "get_query_plan",
    "get_slot_plan",
    "slot_plan_to_query_plan",
    "detect_owner_from_starter",
    "reset_planner_chain",
    "STARTER_OWNER_MAP",
]
