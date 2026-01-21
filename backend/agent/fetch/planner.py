"""SQL Sorcerer-style query planner - generates SQL directly."""

import logging
from datetime import datetime

from pydantic import BaseModel, Field

from backend.agent.fetch.sql.schema import get_schema_sql
from backend.core.llm import create_anthropic_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Transform natural language requests into valid DuckDB SQL queries.

Today: {today}

## DATABASE SCHEMA

```sql
{schema}
```

## NOTES
- Each table has a "notes" column containing free-text context (insights, concerns, history)
- Include notes in SELECT when the question asks about qualitative information
- "Recent" or "recently" means within the last 90 days"""

_HUMAN_PROMPT = """User's question: {question}

{conversation_history_section}"""


class SQLPlan(BaseModel):
    """LLM output containing SQL query."""

    sql: str = Field(description="The SQL query to execute")


def _get_planner_chain():
    """Get the planner chain (not cached - system prompt includes dynamic date)."""
    system_prompt = _SYSTEM_PROMPT.format(
        today=datetime.now().strftime("%Y-%m-%d"),
        schema=get_schema_sql(),
    )
    chain = create_anthropic_chain(
        system_prompt=system_prompt,
        human_prompt=_HUMAN_PROMPT,
        structured_output=SQLPlan,
    )
    logger.debug("Created planner chain")
    return chain


def get_sql_plan(
    question: str,
    conversation_history: str = "",
    previous_error: str | None = None,
) -> SQLPlan:
    """
    Get SQL directly from LLM using SQL Sorcerer approach.

    Args:
        question: User's question
        conversation_history: Formatted conversation context
        previous_error: Error from previous query attempt (for retry)

    Returns SQLPlan with SQL string and needs_rag flag.
    """
    # Build conversation history section with optional error context
    history_section = ""
    if conversation_history:
        history_section = f"=== CONVERSATION HISTORY ===\n{conversation_history}\n"
    if previous_error:
        history_section += f"\n[PREVIOUS QUERY FAILED]\n{previous_error}\nPlease fix the query."

    result: SQLPlan = _get_planner_chain().invoke({
        "question": question,
        "conversation_history_section": history_section,
    })
    logger.info("SQL Planner: %s", result.sql[:80])
    return result


__all__ = ["SQLPlan", "get_sql_plan"]
