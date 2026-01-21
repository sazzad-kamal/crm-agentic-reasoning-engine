"""
Minimal DuckDB connection manager with CSV loading.

Loads CSV files with all schema columns including notes.
"""

import logging
import threading
from pathlib import Path

import duckdb

from backend.agent.fetch.sql.schema import get_all_table_columns, get_table_names

logger = logging.getLogger(__name__)

_CSV_PATH = Path(__file__).parent.parent.parent.parent / "data" / "csv"


def _load_csvs(conn: duckdb.DuckDBPyConnection) -> None:
    """Load CSV files into DuckDB with all schema columns."""
    table_columns = get_all_table_columns()
    for table in get_table_names():
        csv_file = _CSV_PATH / f"{table}.csv"
        if csv_file.exists():
            columns = ", ".join(table_columns[table])
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} AS "
                f"SELECT {columns} FROM read_csv_auto('{csv_file.as_posix()}')"
            )
            logger.debug(f"Loaded table '{table}' from {csv_file}")
        else:
            logger.warning(f"CSV file not found: {csv_file}")


_thread_local = threading.local()


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a thread-local DuckDB connection with CSV tables loaded."""
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        _thread_local.conn = duckdb.connect(":memory:")
        _load_csvs(_thread_local.conn)
        logger.debug("Created new DuckDB connection with CSV tables")
    conn: duckdb.DuckDBPyConnection = _thread_local.conn
    return conn


__all__ = ["get_connection"]
