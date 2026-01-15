"""
Minimal DuckDB connection manager with CSV loading.

Replaces the complex CRMDataStore class with direct SQL execution.
"""

import logging
import threading
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

# CSV tables to load
_CSV_TABLES = ["companies", "contacts", "activities", "history", "opportunities"]


def _get_csv_base_path() -> Path:
    """Get the base path for CSV files (data/crm/ or data/csv/)."""
    backend_root = Path(__file__).parent.parent.parent.parent
    preferred = backend_root / "data" / "crm"
    if preferred.exists() and preferred.is_dir():
        return preferred
    return backend_root / "data" / "csv"


def _load_csvs(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Load all CSV files into DuckDB tables."""
    for table in _CSV_TABLES:
        csv_file = csv_path / f"{table}.csv"
        if csv_file.exists():
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} AS "
                f"SELECT * FROM read_csv_auto('{csv_file.as_posix()}')"
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
    return _thread_local.conn


def reset_connection() -> None:
    """Reset the thread-local connection (for testing)."""
    if hasattr(_thread_local, "conn") and _thread_local.conn is not None:
        _thread_local.conn.close()
        _thread_local.conn = None


__all__ = ["get_connection", "reset_connection"]
