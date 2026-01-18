"""SQL Sorcerer-style query planner - generates SQL directly."""

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from backend.agent.fetch.sql.schema import get_schema_sql
from backend.core.llm import load_prompt, parse_json_response

logger = logging.getLogger(__name__)

_ROUTER_MODEL = "claude-sonnet-4-5-20241022"

_DIR = Path(__file__).parent


class SQLPlan(BaseModel):
    """LLM output containing SQL query and RAG flag."""

    sql: str = Field(description="The SQL query to execute")
    needs_rag: bool = Field(default=False, description="Whether RAG context is needed")


@lru_cache
def _get_client() -> anthropic.Anthropic:
    """Get Anthropic client (cached)."""
    return anthropic.Anthropic()


def get_sql_plan(question: str, conversation_history: str = "") -> SQLPlan:
    """
    Get SQL directly from LLM using SQL Sorcerer approach.

    Returns SQLPlan with SQL string and needs_rag flag.
    """
    prompt = load_prompt(_DIR / "prompt.txt").format(
        today=datetime.now().strftime("%Y-%m-%d"),
        schema=get_schema_sql(),
        conversation_history=conversation_history or "",
        question=question,
    )

    response = _get_client().messages.create(
        model=_ROUTER_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": f"{prompt}\n\nQuestion: {question}"}],
    )

    # Parse JSON from response
    block = response.content[0]
    text = block.text if hasattr(block, "text") else ""
    data = parse_json_response(text)
    result = SQLPlan(**data)
    logger.info("SQL Planner: %s (needs_rag=%s)", result.sql[:80], result.needs_rag)
    return result


__all__ = ["SQLPlan", "get_sql_plan"]
