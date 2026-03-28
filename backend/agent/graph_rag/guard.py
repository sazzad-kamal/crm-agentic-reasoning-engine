"""Cypher safety guard for Neo4j queries.

Validates that LLM-generated Cypher is read-only, blocking any
write operations (CREATE, DELETE, SET, MERGE, REMOVE, DROP).
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Write operations that must be blocked in user-generated Cypher
FORBIDDEN_KEYWORDS = {
    "CREATE", "DELETE", "DETACH", "SET", "REMOVE", "MERGE",
    "DROP", "CALL", "FOREACH",
}

# Max results to prevent unbounded queries
MAX_RESULTS = 1000


@dataclass
class CypherGuardResult:
    """Result of Cypher safety validation."""

    is_safe: bool
    cypher: str
    reason: str = ""


def validate_cypher(cypher: str) -> CypherGuardResult:
    """Validate that a Cypher query is safe (read-only).

    Checks:
    1. Reject empty/whitespace queries
    2. Block write operations (CREATE, DELETE, SET, MERGE, etc.)
    3. Auto-add LIMIT if not present

    Returns:
        CypherGuardResult with is_safe flag and cleaned Cypher
    """
    if not cypher or not cypher.strip():
        return CypherGuardResult(is_safe=False, cypher="", reason="Empty query")

    cleaned = cypher.strip()

    # Tokenize and check for forbidden keywords at statement level
    # Use word boundaries to avoid false positives (e.g., "CREATED" in a string)
    for keyword in FORBIDDEN_KEYWORDS:
        pattern = rf"\b{keyword}\b"
        if re.search(pattern, cleaned, re.IGNORECASE):
            logger.warning(f"[Neo4j Guard] Blocked forbidden keyword: {keyword}")
            return CypherGuardResult(
                is_safe=False, cypher=cleaned,
                reason=f"Forbidden write operation: {keyword}",
            )

    # Auto-add LIMIT if not present
    if not re.search(r"\bLIMIT\b", cleaned, re.IGNORECASE):
        cleaned = f"{cleaned}\nLIMIT {MAX_RESULTS}"
        logger.debug(f"[Neo4j Guard] Auto-added LIMIT {MAX_RESULTS}")

    return CypherGuardResult(is_safe=True, cypher=cleaned)


__all__ = ["CypherGuardResult", "validate_cypher", "MAX_RESULTS"]
