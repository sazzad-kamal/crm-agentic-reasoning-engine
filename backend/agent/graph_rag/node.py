"""Graph RAG node for LangGraph workflow.

Handles multi-hop relationship queries by traversing the Neo4j
CRM knowledge graph. Mirrors the fetch node pattern.
"""

import logging

from backend.agent.graph_rag.connection import get_driver
from backend.agent.graph_rag.executor import execute_cypher
from backend.agent.graph_rag.guard import validate_cypher
from backend.agent.graph_rag.planner import get_cypher_plan
from backend.agent.state import AgentState, format_conversation_for_prompt

logger = logging.getLogger(__name__)


def graph_node(state: AgentState) -> dict:
    """Graph RAG node: query Neo4j knowledge graph for relationship data.

    This node handles multi-hop entity relationship questions that
    benefit from graph traversal rather than SQL joins.

    Args:
        state: Current agent state with question

    Returns:
        Updated state with graph query results in sql_results
    """
    print("[Graph] Node entered", flush=True)

    question = state["question"]
    history = format_conversation_for_prompt(state.get("messages", []))

    try:
        # Plan Cypher query
        plan = get_cypher_plan(question, history)

        # Validate safety (read-only)
        guard_result = validate_cypher(plan.cypher)
        if not guard_result.is_safe:
            logger.warning(f"[Graph] Unsafe Cypher blocked: {guard_result.reason}")
            return {
                "sql_results": {
                    "graph_data": [],
                    "error": f"Query blocked: {guard_result.reason}",
                    "_source": "neo4j",
                }
            }

        # Execute against Neo4j
        driver = get_driver()
        records, error = execute_cypher(guard_result.cypher, driver)

        if error:
            logger.error(f"[Graph] Query failed: {error}")
            return {
                "sql_results": {
                    "graph_data": [],
                    "error": error,
                    "_debug": {"cypher": guard_result.cypher},
                    "_source": "neo4j",
                }
            }

        logger.info(f"[Graph] Returned {len(records)} records")
        return {
            "sql_results": {
                "graph_data": records,
                "data": records,  # Compatible with answer node
                "_debug": {
                    "cypher": guard_result.cypher,
                    "explanation": plan.explanation,
                    "row_count": len(records),
                },
                "_source": "neo4j",
            }
        }

    except Exception as e:
        logger.error(f"[Graph] Failed: {e}")
        return {
            "sql_results": {
                "graph_data": [],
                "error": str(e),
                "_source": "neo4j",
            }
        }


__all__ = ["graph_node"]
