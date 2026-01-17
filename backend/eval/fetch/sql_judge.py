"""LLM judge for SQL semantic correctness."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from backend.agent.core.llm import parse_json_response
from backend.eval.shared import is_mock_mode

logger = logging.getLogger(__name__)

# Thread-safe singleton for OpenAI client
_openai_client = None
_client_lock = threading.Lock()

JUDGE_MODEL = "gpt-5.2"

JUDGE_PROMPT = """You are evaluating whether SQL query results correctly answer a user's question about CRM data.

## Question
{question}

## SQL Query
{sql}

## Query Results
{results}

## Your Task
Evaluate whether the results answer the question:
1. Does the data returned actually answer what was asked?
2. Are the results complete (not missing expected data)?
3. Are the results accurate (no incorrect or irrelevant data)?

Consider:
- If asking for a count, does the result have the right number of rows?
- If asking for specific entities, are they present?
- If asking for aggregations, are the values reasonable?
- If asking for filtered data, is the filter correctly applied?

## Response Format
Respond with JSON only:
{{"passed": true/false, "reasoning": "Brief explanation", "errors": ["error1", "error2"]}}

If passed=true, errors should be an empty list.
If passed=false, list specific issues found."""


def _get_openai_client() -> Any:
    """Get shared OpenAI client (thread-safe singleton)."""
    global _openai_client
    if _openai_client is None:
        with _client_lock:
            if _openai_client is None:
                from openai import OpenAI
                logger.info(f"Initializing SQL Judge LLM ({JUDGE_MODEL})")
                _openai_client = OpenAI()
    return _openai_client


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
    sql_results: dict,
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
    # Mock mode for testing without API
    if is_mock_mode():
        logger.debug("SQL Judge: mock mode - returning pass")
        return True, []

    try:
        client = _get_openai_client()

        prompt = JUDGE_PROMPT.format(
            question=question,
            sql=sql or "No SQL provided",
            results=_format_results(sql_results),
        )

        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0,
        )

        # Parse response
        content = response.choices[0].message.content or ""

        try:
            data = parse_json_response(content)
            passed = bool(data.get("passed", False))
            errors = data.get("errors", [])
            reasoning = data.get("reasoning", "")

            if not passed and reasoning and not errors:
                errors = [reasoning]

            logger.debug(f"SQL Judge: passed={passed}, reasoning={reasoning[:100]}")
            return passed, errors if isinstance(errors, list) else [str(errors)]

        except (json.JSONDecodeError, ValueError):
            logger.warning(f"SQL Judge: failed to parse JSON response: {content[:200]}")
            return False, ["Judge failed to return valid JSON"]

    except Exception as e:
        logger.warning(f"SQL Judge error: {e}")
        # On error, don't fail the eval - just skip validation
        return True, []


__all__ = ["judge_sql_results"]
