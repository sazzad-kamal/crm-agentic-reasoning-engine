"""SQL Sorcerer-style query planner - generates SQL directly."""

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from backend.agent.sql.schema import get_schema_sql
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
- "Tell me more about [company]" is a broad overview question - use UNION ALL to fetch the company row, its opportunities, activities, and history in one result set
- When the user references a company from conversation history with pronouns like "their" (e.g., "their key contact"), resolve the pronoun to that company and use UNION ALL to fetch company + activity + contact data
- CRITICAL: All UNION ALL queries MUST start with a company SELECT using these exact column aliases: source, name, plan, status, health_status, key_date, notes. Subsequent SELECTs must match these 7 columns positionally. Follow the examples exactly.
- "Recent" or "recently" means within the last 90 days
- "Pipeline" = opportunities NOT IN ('Closed Won', 'Closed Lost')
- Use CURRENT_DATE for relative date calculations, never hardcode dates
- Only use columns that exist in the schema
- Resolve pronouns ("their", "them", "this company") using conversation history before building the query
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
SELECT * FROM companies WHERE renewal_date >= CURRENT_DATE AND status = 'Active' ORDER BY renewal_date

Q: "What are the largest deals in the pipeline?"
SELECT * FROM opportunities WHERE stage NOT IN ('Closed Won', 'Closed Lost') ORDER BY amount DESC

Q: "How are deals distributed across stages?"
SELECT stage, COUNT(*) AS deal_count FROM opportunities WHERE stage NOT IN ('Closed Won', 'Closed Lost') GROUP BY stage ORDER BY deal_count DESC

Q: "Who owns the most pipeline value?"
SELECT owner, SUM(amount) AS total_value FROM opportunities WHERE stage NOT IN ('Closed Won', 'Closed Lost') GROUP BY owner ORDER BY total_value DESC

Q: "Which renewals are at risk?"
SELECT * FROM companies WHERE health_status LIKE '%at-risk%' AND renewal_date IS NOT NULL

Q: "Tell me more about Crown Foods"
SELECT 'company' AS source, name, plan, status, health_status, renewal_date::TEXT AS key_date, notes FROM companies WHERE name = 'Crown Foods'
UNION ALL
SELECT 'opportunity', name, stage, type, amount::TEXT, expected_close_date::TEXT, notes FROM opportunities WHERE company_id = (SELECT company_id FROM companies WHERE name = 'Crown Foods')
UNION ALL
SELECT 'activity', type, subject, status, priority, due_date::TEXT, notes FROM activities WHERE company_id = (SELECT company_id FROM companies WHERE name = 'Crown Foods')
UNION ALL
SELECT 'history', type, subject, '', '', occurred_at::TEXT, notes FROM history WHERE company_id = (SELECT company_id FROM companies WHERE name = 'Crown Foods') ORDER BY key_date DESC

Q: "What tasks are due this week?"
SELECT * FROM activities WHERE due_date >= date_trunc('week', CURRENT_DATE) AND due_date < date_trunc('week', CURRENT_DATE) + INTERVAL '7 days' AND status = 'Open'

Q: "What meetings are coming up?"
SELECT * FROM activities WHERE type = 'Meeting' AND status = 'Open' ORDER BY due_date

Q: "Any activity with their key contact this week?" (conversation context: Crown Foods)
SELECT 'company' AS source, name, plan, status, health_status, renewal_date::TEXT AS key_date, notes FROM companies WHERE name = 'Crown Foods'
UNION ALL
SELECT 'activity', type, subject, status, priority, due_date::TEXT, notes FROM activities WHERE company_id = (SELECT company_id FROM companies WHERE name = 'Crown Foods') AND contact_id IN (SELECT contact_id FROM contacts WHERE company_id = (SELECT company_id FROM companies WHERE name = 'Crown Foods') AND role = 'Decision Maker') AND due_date >= date_trunc('week', CURRENT_DATE) AND due_date < date_trunc('week', CURRENT_DATE) + INTERVAL '7 days' AND status = 'Open'
UNION ALL
SELECT 'contact', first_name || ' ' || last_name, role, job_title, email, '', notes FROM contacts WHERE company_id = (SELECT company_id FROM companies WHERE name = 'Crown Foods') AND role = 'Decision Maker'

Q: "Any activity with Crown Foods' key contact this week?"
SELECT 'company' AS source, name, plan, status, health_status, renewal_date::TEXT AS key_date, notes FROM companies WHERE name = 'Crown Foods'
UNION ALL
SELECT 'activity', type, subject, status, priority, due_date::TEXT, notes FROM activities WHERE company_id = (SELECT company_id FROM companies WHERE name = 'Crown Foods') AND contact_id IN (SELECT contact_id FROM contacts WHERE company_id = (SELECT company_id FROM companies WHERE name = 'Crown Foods') AND role = 'Decision Maker') AND due_date >= date_trunc('week', CURRENT_DATE) AND due_date < date_trunc('week', CURRENT_DATE) + INTERVAL '7 days' AND status = 'Open'
UNION ALL
SELECT 'contact', first_name || ' ' || last_name, role, job_title, email, '', notes FROM contacts WHERE company_id = (SELECT company_id FROM companies WHERE name = 'Crown Foods') AND role = 'Decision Maker'"""

_HUMAN_PROMPT = """User's question: {question}

{conversation_history_section}"""


class SQLPlan(BaseModel):
    """LLM output containing SQL query."""

    sql: str = Field(description="The SQL query to execute")


def _get_planner_chain() -> Any:
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
