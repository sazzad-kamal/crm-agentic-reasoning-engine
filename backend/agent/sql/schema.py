"""Schema loader for CRM database."""

from functools import cache
from pathlib import Path

import yaml

_SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


@cache
def _load_schema() -> dict:
    """Load schema from YAML file (cached)."""
    with open(_SCHEMA_PATH) as f:
        result: dict = yaml.safe_load(f)
        return result


def get_table_names() -> list[str]:
    """Get list of table names."""
    return list(_load_schema()["tables"].keys())


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


__all__ = ["get_table_names", "get_all_table_columns", "get_schema_sql"]
