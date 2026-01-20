"""SQL Sorcerer-style query planner - generates SQL directly."""

import logging
from datetime import datetime

from pydantic import BaseModel, Field

from backend.agent.fetch.rag.schema import get_rag_schema
from backend.agent.fetch.sql.schema import get_schema_sql
from backend.core.llm import create_anthropic_chain

logger = logging.getLogger(__name__)

_HUMAN_PROMPT = "{question}"

_SYSTEM_PROMPT = """Transform natural language requests into valid DuckDB SQL queries and decide if RAG context is needed.

Today: {today}

## DATABASE SCHEMA

```sql
{schema}
```

## RAG KNOWLEDGE BASE
Additional context available via RAG search (not in SQL tables).

Decision:
- needs_rag=true: "Why is this deal stuck?", "What are their concerns?", "How should I approach them?"
- needs_rag=false: "How many opportunities?", "What's the close date?", "List contacts at X"

Available context:
{rag_schema}

## NOTES
- "Recent" or "recently" means within the last 90 days

## CONVERSATION HISTORY
{conversation_history}"""


class SQLPlan(BaseModel):
    """LLM output containing SQL query and RAG flag."""

    sql: str = Field(description="The SQL query to execute")
    needs_rag: bool = Field(default=False, description="Whether RAG context is needed")


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
    error_context = ""
    if previous_error:
        error_context = f"\n\n[PREVIOUS QUERY FAILED]\n{previous_error}\nPlease fix the query."

    system_prompt = _SYSTEM_PROMPT.format(
        today=datetime.now().strftime("%Y-%m-%d"),
        schema=get_schema_sql(),
        rag_schema=get_rag_schema(),
        conversation_history=(conversation_history or "") + error_context,
    )

    chain = create_anthropic_chain(
        system_prompt=system_prompt,
        human_prompt=_HUMAN_PROMPT,
        structured_output=SQLPlan,
    )
    result: SQLPlan = chain.invoke({"question": question})
    logger.info("SQL Planner: %s (needs_rag=%s)", result.sql[:80], result.needs_rag)
    return result


__all__ = ["SQLPlan", "get_sql_plan"]
