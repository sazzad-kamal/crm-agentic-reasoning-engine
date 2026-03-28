"""Graph schema loader for Neo4j CRM knowledge graph."""

from functools import cache
from pathlib import Path

import yaml

_SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


@cache
def _load_schema() -> dict:
    """Load graph schema from YAML file (cached)."""
    with open(_SCHEMA_PATH) as f:
        result: dict = yaml.safe_load(f)
        return result


def get_graph_schema_prompt() -> str:
    """Render the graph schema as a string for the LLM Cypher-generation prompt."""
    schema = _load_schema()
    parts = ["## Node Labels and Properties\n"]

    for label, config in schema["nodes"].items():
        props = ", ".join(config["properties"])
        parts.append(f"(:{label} {{{props}}})")

    parts.append("\n## Relationships\n")
    for rel in schema["relationships"]:
        parts.append(f"(:{rel['from']})-[:{rel['type']}]->(:{rel['to']})")

    return "\n".join(parts)


__all__ = ["get_graph_schema_prompt"]
