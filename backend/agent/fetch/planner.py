"""SQL Sorcerer-style query planner - generates SQL directly."""

import logging
from datetime import datetime

from pydantic import BaseModel, Field

from backend.agent.fetch.sql.schema import get_schema_sql
from backend.core.llm import create_anthropic_chain

logger = logging.getLogger(__name__)

_HUMAN_PROMPT = "{question}"

_SYSTEM_PROMPT = """Transform natural language requests into valid DuckDB SQL queries.

Today: {today}

## DATABASE SCHEMA

```sql
{schema}
```

## NOTES
- "Recent" or "recently" means within the last 90 days

## RAG KNOWLEDGE BASE
The following private text is available via RAG search (not in SQL tables). Set needs_rag=true when the question requires this context:

| Entity | Contains | Common Use Cases |
|--------|----------|------------------|
| company | Key contacts, decision dynamics, adoption status, renewal concerns, win-back notes, attached docs | "Why is X at risk?", "What's the background on X?", "How do we approach renewal?" |
| contact | Communication preferences, concerns, objections, influence, technical requirements | "How should I approach Beth?", "What are Joe's concerns?", "Who is the champion?" |
| opportunity | Deal risks, blockers, recommended next steps, dependencies, proposals/contracts | "What's blocking this deal?", "Why is this stuck?", "What should I do next?" |
| activity | Call/meeting notes with context, concerns raised, action items, prep notes | "What was discussed in the call?", "What did we agree on?", "Any prep notes?" |
| history | Past interaction summaries, outcomes, what was communicated | "What happened last time?", "Any previous discussions?", "Email history?" |

Set needs_rag=false for pure data queries (counts, lists, values, dates, stages).

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
