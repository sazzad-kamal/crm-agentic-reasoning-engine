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
- For qualitative questions (why, details, concerns), the answer is often in the primary entity's notes - don't add extra JOINs unless you are certain that you need it
- "Recent" or "recently" means within the last 90 days
- "Pipeline" = opportunities NOT IN ('Closed Won', 'Closed Lost')
- Use CURRENT_DATE for relative date calculations, never hardcode dates
- Only use columns that exist in the schema
- CRITICAL: Only filter by values the user is asking to filter by, not contextual references
- No exclusion filters (NOT IN, IS NOT NULL) unless asked
- CRITICAL: Minimal JOINs - query only the primary table unless a JOIN is essential for the answer
- Use INNER JOIN by default; LEFT JOIN only for optional relationships
- Never use LIMIT unless the user asks for a specific number (e.g., "top 3", "first 5")

## TABLE DISTINCTIONS
- activities: future/scheduled tasks (activity_id, due_date, completed_at)
- history: past interactions (history_id, occurred_at) - DO NOT use activity_id on history
- Company names (e.g., "Beta Tech") are in companies.name, NOT in opportunities.name
- health_status uses hyphenated values like 'at-risk-low-activity' - use LIKE '%at-risk%'

## EXAMPLES

Q: "What deals are in the pipeline?"
SELECT * FROM opportunities WHERE stage NOT IN ('Closed Won', 'Closed Lost')

Q: "Which accounts are up for renewal?"
SELECT * FROM companies WHERE renewal_date >= CURRENT_DATE ORDER BY renewal_date

Q: "Who are the contacts at Delta Health?"
SELECT c.* FROM contacts c JOIN companies co ON c.company_id = co.company_id WHERE co.name = 'Delta Health Clinics'

Q: "Which deals are closest to closing?"
SELECT * FROM opportunities WHERE stage NOT IN ('Closed Won', 'Closed Lost') ORDER BY expected_close_date ASC

Q: "What stages have the most deals?"
SELECT stage, COUNT(*) AS deal_count FROM opportunities GROUP BY stage ORDER BY deal_count DESC

Q: "Who owns the most pipeline value?"
SELECT owner, SUM(amount) AS total_value FROM opportunities WHERE stage NOT IN ('Closed Won', 'Closed Lost') GROUP BY owner ORDER BY total_value DESC

Q: "Which renewals are at risk?"
SELECT * FROM companies WHERE health_status LIKE '%at-risk%' AND renewal_date IS NOT NULL

Q: "What's the history with Crown Foods?"
SELECT h.* FROM history h JOIN companies c ON h.company_id = c.company_id WHERE c.name = 'Crown Foods'"""

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
    """Generate SQL from natural language question."""
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
