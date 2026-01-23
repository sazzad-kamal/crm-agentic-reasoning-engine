"""LLM judge for SQL semantic correctness."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain

logger = logging.getLogger(__name__)


class JudgeResult(BaseModel):
    """Structured output from the SQL judge."""

    passed: bool = Field(description="Whether the results correctly answer the question")
    errors: list[str] = Field(default_factory=list, description="List of issues found (required if passed=false)")

_SYSTEM_PROMPT = """You are evaluating whether SQL query results correctly answer a user's question about CRM data.

Evaluate whether the results answer the question:
1. Does the data returned actually answer what was asked?
2. Are the results complete (not missing expected data)?
3. Are the results accurate (no incorrect or irrelevant data)?

Consider:
- If asking for a count, does the result have the right number of rows?
- If asking for specific entities, are they present?
- If asking for aggregations, are the values reasonable?
- If asking for filtered data, is the filter correctly applied?

If passed=true, errors must be an empty list.
If passed=false, errors must list specific issues found (required)."""

_HUMAN_PROMPT = """## Question
{question}

## SQL Query
{sql}

## Query Results
{results}"""


def _format_results(sql_results: dict) -> str:
    """Format SQL results for the prompt."""
    if not sql_results:
        return "No results returned"

    # Truncate if too large
    result_str = json.dumps(sql_results, indent=2, default=str)
    if len(result_str) > 4000:
        result_str = result_str[:4000] + "\n... (truncated)"
    return result_str


def judge_sql_results(
    question: str,
    sql: str,
    sql_results: dict[str, list],
) -> tuple[bool, list[str]]:
    """
    Use LLM to judge if SQL results correctly answer the question.

    Args:
        question: The user's question
        sql: The SQL query that was executed
        sql_results: Dict of query results (column name -> list of values)

    Returns:
        Tuple of (passed: bool, errors: list[str])
        - passed: True if the results correctly answer the question
        - errors: List of issues found (empty if passed)
    """
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=512,
        structured_output=JudgeResult,
        streaming=False,
    )

    try:
        result: JudgeResult = chain.invoke({
            "question": question,
            "sql": sql or "No SQL provided",
            "results": _format_results(sql_results),
        })

        logger.debug(f"SQL Judge: passed={result.passed}, errors={result.errors}")
        return result.passed, result.errors

    except Exception as e:
        logger.warning(f"SQL Judge error: {e}")
        return False, [f"Judge API error: {e}"]


__all__ = ["judge_sql_results"]
