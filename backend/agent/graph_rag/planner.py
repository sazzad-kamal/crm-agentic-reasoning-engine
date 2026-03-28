"""Cypher query planner for Neo4j graph queries.

Generates Cypher from natural language using the graph schema,
mirroring the SQL planner pattern in fetch/planner.py.
"""

import logging
from datetime import datetime

from pydantic import BaseModel, Field

from backend.agent.graph_rag.schema import get_graph_schema_prompt
from backend.core.llm import create_anthropic_chain

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Generate a valid Neo4j Cypher query for the given CRM question.

Today: {today}

## GRAPH SCHEMA

{schema}

## RULES
- Use MATCH patterns to traverse relationships
- Return meaningful properties, not entire nodes
- Use WHERE for filtering
- Use OPTIONAL MATCH for relationships that may not exist
- For multi-hop queries, chain MATCH patterns: (a)-[:REL1]->(b)-[:REL2]->(c)
- Use COLLECT() for aggregating related items
- Company names are case-sensitive — use toLower() for matching
- "at risk" health status uses values like 'at-risk-low-activity' — use CONTAINS 'at-risk'
- "Pipeline" = opportunities WHERE stage is NOT 'Closed Won' and NOT 'Closed Lost'
- Never use write operations (CREATE, DELETE, SET, MERGE)
- Do NOT include LIMIT — it will be added automatically

## EXAMPLES

Q: "Which contacts at at-risk companies have deals closing this month?"
MATCH (c:Company)-[:HAS_CONTACT]->(ct:Contact),
      (c)-[:HAS_OPPORTUNITY]->(o:Opportunity)
WHERE c.health_status CONTAINS 'at-risk'
  AND o.expected_close_date >= date().toString()
RETURN ct.first_name, ct.last_name, c.name AS company, o.name AS deal, o.amount, o.expected_close_date

Q: "Show the full relationship chain for Acme Manufacturing"
MATCH (c:Company {{name: 'Acme Manufacturing'}})
OPTIONAL MATCH (c)-[:HAS_CONTACT]->(ct:Contact)
OPTIONAL MATCH (c)-[:HAS_OPPORTUNITY]->(o:Opportunity)
OPTIONAL MATCH (c)-[:HAS_ACTIVITY]->(a:Activity)
RETURN c.name, COLLECT(DISTINCT ct.first_name + ' ' + ct.last_name) AS contacts,
       COLLECT(DISTINCT o.name) AS opportunities, COLLECT(DISTINCT a.subject) AS activities
"""

_HUMAN_PROMPT = """Question: {question}

Conversation history:
{history}"""


class CypherPlan(BaseModel):
    """Planned Cypher query with explanation."""

    cypher: str = Field(description="The Cypher query to execute")
    explanation: str = Field(description="Brief explanation of the query strategy")


def get_cypher_plan(question: str, conversation_history: str = "") -> CypherPlan:
    """Generate a Cypher query plan from a natural language question.

    Args:
        question: User's question about CRM relationships
        conversation_history: Previous conversation context

    Returns:
        CypherPlan with the query and explanation
    """
    chain = create_anthropic_chain(
        system_prompt=_SYSTEM_PROMPT.format(
            today=datetime.now().strftime("%Y-%m-%d"),
            schema=get_graph_schema_prompt(),
        ),
        human_prompt=_HUMAN_PROMPT,
        structured_output=CypherPlan,
    )

    result: CypherPlan = chain.invoke({
        "question": question,
        "history": conversation_history or "No previous conversation",
    })

    logger.info(f"[Neo4j Planner] Generated Cypher: {result.cypher[:100]}...")
    return result


__all__ = ["CypherPlan", "get_cypher_plan"]
