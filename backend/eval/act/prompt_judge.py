"""Static validation of demo question configurations.

Validates that for each demo question:
1. The API fetch strategy is correct for the question
2. The answer guidance will produce good answers
3. The action guidance will suggest appropriate actions

Run with: python -m backend.eval.act.prompt_judge
"""

from __future__ import annotations

import inspect
import re

from pydantic import BaseModel, Field

from backend.core.llm import create_openai_chain
from backend.eval.act.schema import ACT_API_SCHEMA

# SLO thresholds
SLO_FETCH_SCORE = 0.75
SLO_FETCH_LATENCY_P90_MS = 20000  # P90 < 20s for demo (allows occasional retries)
SLO_FETCH_SUCCESS_RATE = 0.95  # 95% of calls must succeed
SLO_ANSWER_SCORE = 0.90
SLO_ACTION_SCORE = 0.90


class QuestionConfigResult(BaseModel):
    """Result from validating a single question's configuration."""

    fetch_score: float = Field(
        ge=0.0, le=1.0,
        description="Does the API call correctly fetch data needed for this question?"
    )
    fetch_issue: str = Field(description="Main issue with fetch strategy, or 'none'")

    answer_score: float = Field(
        ge=0.0, le=1.0,
        description="Does the answer guidance help produce a focused, data-grounded answer?"
    )
    answer_issue: str = Field(description="Main issue with answer guidance, or 'none'")

    action_score: float = Field(
        ge=0.0, le=1.0,
        description="Does the action guidance lead to specific, CRM-appropriate actions?"
    )
    action_issue: str = Field(description="Main issue with action guidance, or 'none'")


def _extract_fetch_code(question: str) -> str:
    """Extract the actual fetch code for a question from act_fetch.py INCLUDING the return statement."""
    from backend import act_fetch

    source = inspect.getsource(act_fetch.act_fetch)

    # Find the code block for this question INCLUDING its return statement
    # Pattern: if q == "question": ... return {...}, "error": None}
    escaped_q = re.escape(question)
    # Match from the if/elif to the return statement (inclusive)
    pattern = rf'(?:if|elif) q == "{escaped_q}":(.*?return \{{"data":.*?"error": None\}})'

    match = re.search(pattern, source, re.DOTALL)
    if match:
        code_block = match.group(1).strip()
        # Clean up indentation
        lines = code_block.split('\n')
        if lines:
            # Find minimum indentation
            min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
            code_block = '\n'.join(line[min_indent:] if len(line) > min_indent else line for line in lines)
        return code_block[:8000]  # Allow full code including return for complex questions

    return "Code not found"


_SYSTEM_PROMPT = """You are validating Act! CRM demo configurations. Be critical but fair.

{schema}

## Architecture

The system has 3 layers:
1. **Fetch**: Makes Act! API calls to get raw data (contacts, opportunities, history, etc.)
2. **DuckDB Processing**: After fetch, data is loaded into DuckDB for SQL operations:
   - JOINs across datasets (e.g., opportunities JOIN contacts JOIN history)
   - Aggregations (GROUP BY, SUM, COUNT, AVG)
   - Window functions (ROW_NUMBER, LAG for trends)
   - Complex filtering with date math
   - The Python code in fetch shows these operations inline (dict comprehensions, loops)
3. **LLM**: Uses processed data + guidance to generate answer/action

When evaluating fetch code, consider that Python dict operations and loops ARE the DuckDB-equivalent
processing. Comments like "=== DUCKDB: ..." describe what SQL operations the Python code performs.

## Common Variables (defined at function level, available to all question blocks)
- `today = time.strftime("%Y-%m-%d", time.gmtime())` - current UTC date string
- `q = question.strip()` - the question being processed
These variables are ALWAYS available even if not shown in the extracted code block.

## Scoring Guide

### Fetch (0.0-1.0):
- 1.0: Fetches correct endpoints AND processes/joins data correctly for the question
- 0.8: Good fetch + processing with minor gaps (e.g., could add one more field)
- 0.6: Fetches right data but processing logic has issues
- 0.4: Missing important data sources or broken joins
- 0.0: Wrong endpoints or will fail

### Answer Guidance (0.0-1.0):
- 1.0: Specific focus areas that match the PROCESSED data (post-DuckDB)
- 0.8: Good direction, matches most processed output fields
- 0.6: Partially matches processed data
- 0.4: Too vague or references fields not in processed output
- 0.0: Misleading or contradicts the data

### Action Guidance (0.0-1.0):
- 1.0: Will produce specific actions with who/what/when using processed data
- 0.8: Good specificity, minor gaps
- 0.6: Has who/what/when but could be more concrete
- 0.4: Too generic ("follow up" without details)
- 0.0: Inappropriate for CRM context

For each issue field: state the specific problem, or "none" if no issues."""

_HUMAN_PROMPT = """## Question
"{question}"

## Actual Fetch Code (from act_fetch.py)
```python
{fetch_code}
```

## DuckDB Processing (happens after fetch, before LLM)
The fetch code includes inline Python that performs these DuckDB-equivalent operations:
- Dict comprehensions and loops = JOIN operations across fetched datasets
- Filtering with date comparisons = WHERE clauses with date math
- Aggregations in loops (sum, count, max) = GROUP BY with aggregate functions
- Sorting = ORDER BY
- Comments starting with "=== DUCKDB:" describe the SQL-equivalent operations

The return statement shows the FINAL processed output that the LLM will use.

## Answer Guidance
"{answer_guidance}"

## Action Guidance
"{action_guidance}"

Score this configuration. Remember: the Python code IS the processing layer - evaluate based on what the processed output contains, not just the raw API responses."""


def judge_question_config(
    question: str,
    answer_guidance: str,
    action_guidance: str,
) -> QuestionConfigResult:
    """Validate a demo question's configuration using actual code."""

    fetch_code = _extract_fetch_code(question)

    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT.format(schema=ACT_API_SCHEMA),
        human_prompt=_HUMAN_PROMPT,
        max_tokens=1024,  # Enough for detailed issue explanations
        structured_output=QuestionConfigResult,
        streaming=False,
        model="gpt-5.2-pro",  # Use best model for critical static validation
    )

    result: QuestionConfigResult = chain.invoke({
        "question": question,
        "fetch_code": fetch_code,
        "answer_guidance": answer_guidance,
        "action_guidance": action_guidance,
    })

    return result


def validate_all_demo_questions() -> dict[str, QuestionConfigResult]:
    """Validate all 5 demo question configurations."""
    from backend.act_fetch import DEMO_PROMPTS, DEMO_STARTERS

    results = {}

    for question in DEMO_STARTERS:
        prompts = DEMO_PROMPTS.get(question, {})

        result = judge_question_config(
            question=question,
            answer_guidance=prompts.get("answer", ""),
            action_guidance=prompts.get("action", ""),
        )
        results[question] = result

    return results


def _sanitize(text: str) -> str:
    """Remove non-ASCII chars for console output."""
    return text.encode("ascii", "replace").decode("ascii")


def main() -> None:
    """Run static validation on all demo questions."""
    print("=" * 70)
    print("Demo Question Configuration Validation")
    print("=" * 70)

    results = validate_all_demo_questions()

    all_passed = True
    for i, (question, result) in enumerate(results.items(), 1):
        passed = (
            result.fetch_score >= SLO_FETCH_SCORE
            and result.answer_score >= SLO_ANSWER_SCORE
            and result.action_score >= SLO_ACTION_SCORE
        )
        status = "[PASS]" if passed else "[FAIL]"
        all_passed = all_passed and passed

        print(f"\n{i}. {status} \"{question}\"")
        print(f"   Fetch:  {result.fetch_score:.2f}", end="")
        if result.fetch_issue != "none":
            print(f" - {_sanitize(result.fetch_issue)}")
        else:
            print()

        print(f"   Answer: {result.answer_score:.2f}", end="")
        if result.answer_issue != "none":
            print(f" - {_sanitize(result.answer_issue)}")
        else:
            print()

        print(f"   Action: {result.action_score:.2f}", end="")
        if result.action_issue != "none":
            print(f" - {_sanitize(result.action_issue)}")
        else:
            print()

    print("\n" + "=" * 70)
    passed_count = sum(
        1 for r in results.values()
        if r.fetch_score >= SLO_FETCH_SCORE
        and r.answer_score >= SLO_ANSWER_SCORE
        and r.action_score >= SLO_ACTION_SCORE
    )
    print(f"Summary: {passed_count}/{len(results)} questions passed (SLO >= {SLO_FETCH_SCORE})")

    # Calculate averages
    avg_fetch = sum(r.fetch_score for r in results.values()) / len(results)
    avg_answer = sum(r.answer_score for r in results.values()) / len(results)
    avg_action = sum(r.action_score for r in results.values()) / len(results)

    print(f"Averages: Fetch={avg_fetch:.2f} Answer={avg_answer:.2f} Action={avg_action:.2f}")

    if not all_passed:
        print("\n[WARN] Review issues above and update configurations")


if __name__ == "__main__":
    main()
