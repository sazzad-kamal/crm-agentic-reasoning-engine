"""
Schema loader - single source of truth for CRM database schema.

Loads schema.yaml and provides:
- get_table_columns(): For connection.py views
- get_schema_sql(): For prompt.txt LLM context
"""

from functools import lru_cache
from pathlib import Path

import yaml

_SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


@lru_cache
def _load_schema() -> dict:
    """Load schema from YAML file (cached)."""
    with open(_SCHEMA_PATH) as f:
        return yaml.safe_load(f)


def get_table_names() -> list[str]:
    """Get list of table names."""
    return list(_load_schema()["tables"].keys())


def get_table_columns(table: str) -> list[str]:
    """Get column names for a table (excludes notes)."""
    return list(_load_schema()["tables"][table]["columns"].keys())


def get_all_table_columns() -> dict[str, list[str]]:
    """Get all table columns as dict (for connection.py)."""
    schema = _load_schema()
    return {
        table: list(config["columns"].keys())
        for table, config in schema["tables"].items()
    }


def get_schema_sql() -> str:
    """Generate SQL CREATE TABLE statements for LLM prompt."""
    schema = _load_schema()
    statements = []

    for table, config in schema["tables"].items():
        columns = config["columns"]
        col_defs = [f"    {col} {dtype}" for col, dtype in columns.items()]
        stmt = f"CREATE TABLE {table} (\n" + ",\n".join(col_defs) + "\n);"
        statements.append(stmt)

    return "\n\n".join(statements)


__all__ = ["get_table_names", "get_table_columns", "get_all_table_columns", "get_schema_sql"]
