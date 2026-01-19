"""LLM judge for SQL semantic correctness."""

from __future__ import annotations

import json
import logging

from backend.core.llm import call_openai_json

logger = logging.getLogger(__name__)

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
    prompt = JUDGE_PROMPT.format(
        question=question,
        sql=sql or "No SQL provided",
        results=_format_results(sql_results),
    )

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            data = call_openai_json(prompt)
            passed = bool(data.get("passed", False))
            errors = data.get("errors", [])
            reasoning = data.get("reasoning", "")

            if not passed and reasoning and not errors:
                errors = [reasoning]

            logger.debug(f"SQL Judge: passed={passed}, reasoning={reasoning[:100]}")
            return passed, errors if isinstance(errors, list) else [str(errors)]

        except json.JSONDecodeError:
            logger.warning("SQL Judge: failed to parse JSON response")
            return False, ["Judge failed to return valid JSON"]

        except Exception as e:
            last_error = e
            if attempt == 0:
                logger.debug(f"SQL Judge retry after error: {e}")

    logger.warning(f"SQL Judge error after retries: {last_error}")
    return False, [f"Judge API error: {last_error}"]


__all__ = ["judge_sql_results"]
