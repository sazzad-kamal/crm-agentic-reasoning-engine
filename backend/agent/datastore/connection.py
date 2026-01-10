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
CSV_TABLES = {
    "companies": "companies.csv",
    "contacts": "contacts.csv",
    "activities": "activities.csv",
    "history": "history.csv",
    "opportunities": "opportunities.csv",
    "attachments": "attachments.csv",
}


def get_csv_base_path() -> Path:
    """
    Get the base path for CSV files with fallback logic.

    Priority:
    1. data/crm/ (if exists)
    2. data/csv/ (fallback)
    """
    backend_root = Path(__file__).parent.parent.parent

    preferred = backend_root / "data" / "crm"
    if preferred.exists() and preferred.is_dir():
        return preferred

    # Fallback to csv directory (always exists in project)
    return backend_root / "data" / "csv"


def _load_csvs(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Load all CSV files into DuckDB tables."""
    for table_name, filename in CSV_TABLES.items():
        csv_file = csv_path / filename
        if csv_file.exists():
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} AS
                SELECT * FROM read_csv_auto('{csv_file.as_posix()}')
                """
            )
            logger.debug(f"Loaded table '{table_name}' from {csv_file}")
        else:
            logger.warning(f"CSV file not found: {csv_file}")


# Thread-local connection storage
_thread_local = threading.local()


def get_connection(csv_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """
    Get a thread-local DuckDB connection with CSV tables loaded.

    The connection is created once per thread and reused.

    Args:
        csv_path: Optional custom path to CSV files

    Returns:
        DuckDB connection with all tables loaded
    """
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        _thread_local.conn = duckdb.connect(":memory:")
        path = csv_path or get_csv_base_path()
        _load_csvs(_thread_local.conn, path)
        logger.debug("Created new DuckDB connection with CSV tables")

    conn: duckdb.DuckDBPyConnection = _thread_local.conn
    return conn


def reset_connection() -> None:
    """Reset the thread-local connection (for testing)."""
    if hasattr(_thread_local, "conn") and _thread_local.conn is not None:
        _thread_local.conn.close()
        _thread_local.conn = None
        logger.debug("Reset DuckDB connection")


def close_connection() -> None:
    """Close the thread-local connection."""
    reset_connection()


__all__ = [
    "get_connection",
    "reset_connection",
    "close_connection",
    "get_csv_base_path",
    "CSV_TABLES",
]
