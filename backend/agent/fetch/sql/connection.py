"""
Minimal DuckDB connection manager with CSV loading.

Replaces the complex CRMDataStore class with direct SQL execution.
Creates views using schema from schema.yaml (excludes notes columns).
"""

import logging
import threading
from pathlib import Path

import duckdb

from backend.agent.fetch.schema import get_all_table_columns, get_table_names

logger = logging.getLogger(__name__)


def _get_csv_base_path() -> Path:
    """Get the base path for CSV files (data/crm/ or data/csv/)."""
    backend_root = Path(__file__).parent.parent.parent.parent
    preferred = backend_root / "data" / "crm"
    if preferred.exists() and preferred.is_dir():
        return preferred
    return backend_root / "data" / "csv"


def _load_csvs(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Load all CSV files into DuckDB and create views with schema columns only."""
    table_columns = get_all_table_columns()
    for table in get_table_names():
        csv_file = csv_path / f"{table}.csv"
        if csv_file.exists():
            # Load full CSV into raw table
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table}_raw AS "
                f"SELECT * FROM read_csv_auto('{csv_file.as_posix()}')"
            )
            # Create view with only schema columns (excludes notes)
            columns = ", ".join(table_columns[table])
            conn.execute(
                f"CREATE VIEW IF NOT EXISTS {table} AS "
                f"SELECT {columns} FROM {table}_raw"
            )
            logger.debug(f"Loaded table '{table}' from {csv_file}")
        else:
            logger.warning(f"CSV file not found: {csv_file}")


# Thread-local connection storage
_thread_local = threading.local()


def get_connection(csv_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """
    Get a thread-local DuckDB connection with CSV tables loaded.

    The connection is created once per thread and reused.
    """
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        _thread_local.conn = duckdb.connect(":memory:")
        _load_csvs(_thread_local.conn, csv_path or _get_csv_base_path())
        logger.debug("Created new DuckDB connection with CSV tables")
    conn: duckdb.DuckDBPyConnection = _thread_local.conn
    return conn


__all__ = ["get_connection"]
