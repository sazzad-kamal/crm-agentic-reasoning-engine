"""SQL Sorcerer-style query planner - generates SQL directly."""

import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from backend.agent.core.config import get_config
from backend.agent.core.llm import load_prompt
from backend.agent.fetch.schema import get_schema_sql

logger = logging.getLogger(__name__)

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
    config = get_config()

    prompt = load_prompt(_DIR / "prompt.txt").format(
        today=datetime.now().strftime("%Y-%m-%d"),
        schema=get_schema_sql(),
        conversation_history=conversation_history or "",
        question=question,
    )

    response = _get_client().messages.create(
        model=config.router_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": f"{prompt}\n\nQuestion: {question}"}],
    )

    # Parse JSON from response
    block = response.content[0]
    text = block.text if hasattr(block, "text") else ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        # Try to extract JSON from markdown code block
        import re
        if match := re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            data = json.loads(match.group(1))
        else:
            raise ValueError(f"Failed to parse JSON from response: {text[:200]}") from err

    result = SQLPlan(**data)
    logger.info("SQL Planner: %s (needs_rag=%s)", result.sql[:80], result.needs_rag)
    return result


__all__ = ["SQLPlan", "get_sql_plan"]
