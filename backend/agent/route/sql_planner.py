"""SQL Sorcerer-style query planner - generates SQL directly."""

import logging
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from backend.agent.core.config import get_config
from backend.utils.prompt import load_prompt

logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent


class SQLPlan(BaseModel):
    """LLM output containing SQL query and RAG flag."""

    sql: str = Field(description="The SQL query to execute")
    needs_rag: bool = Field(default=False, description="Whether RAG context is needed")


@lru_cache
def _get_client() -> OpenAI:
    """Get OpenAI client (cached)."""
    return OpenAI()


def _extract_sql(response: str) -> str:
    """Extract SQL from response, handling markdown code blocks."""
    # Try to extract from ```sql ... ``` block
    match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to extract from ``` ... ``` block
    match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return as-is if no code blocks
    return response.strip()


def _needs_rag(question: str) -> bool:
    """Determine if question needs RAG context."""
    rag_keywords = ["why", "reason", "concern", "note", "comment", "history", "happening"]
    q_lower = question.lower()
    return any(kw in q_lower for kw in rag_keywords)


def get_sql_plan(question: str, conversation_history: str = "") -> SQLPlan:
    """
    Get SQL directly from LLM using SQL Sorcerer approach.

    Returns SQLPlan with SQL string and needs_rag flag.
    """
    config = get_config()

    prompt = load_prompt(_DIR / "sql_prompt.txt").format(
        today=datetime.now().strftime("%Y-%m-%d"),
        conversation_history=conversation_history or "",
        question=question,
    )

    response = _get_client().chat.completions.create(
        model=config.router_model,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    content = response.choices[0].message.content or ""
    sql = _extract_sql(content)

    result = SQLPlan(
        sql=sql,
        needs_rag=_needs_rag(question),
    )

    logger.info("SQL Planner: %s", sql[:80])
    return result


__all__ = ["SQLPlan", "get_sql_plan"]
