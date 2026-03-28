"""Cypher query executor for Neo4j."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def execute_cypher(cypher: str, driver: Any) -> tuple[list[dict], str | None]:
    """Execute a Cypher query against Neo4j.

    Args:
        cypher: The Cypher query to execute
        driver: Neo4j driver instance

    Returns:
        Tuple of (results as list of dicts, error message or None)
    """
    try:
        with driver.session() as session:
            result = session.run(cypher)
            records = [dict(record) for record in result]
            logger.info(f"[Neo4j] Query returned {len(records)} records")
            return records, None
    except Exception as e:
        logger.error(f"[Neo4j] Query execution failed: {e}")
        return [], str(e)


__all__ = ["execute_cypher"]
