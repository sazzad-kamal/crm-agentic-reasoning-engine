"""
Shared utilities for RAG module.

This module provides common utilities used across the RAG pipeline,
retrieval, and ingestion components.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Get backend root directory
_BACKEND_ROOT = Path(__file__).parent.parent

# Candidate paths for CSV directory (in order of preference)
# The actual data lives at backend/data/csv
CSV_DIR_CANDIDATES = [
    _BACKEND_ROOT / "data" / "csv",      # Primary: backend/data/csv
    _BACKEND_ROOT / "data" / "crm",      # Alternative naming
    Path.cwd() / "backend" / "data" / "csv",  # From project root
]

# Required files - minimal set needed for ingestion
REQUIRED_CSV_FILES = [
    "companies.csv",
    "history.csv",
]


def find_csv_dir() -> Path:
    """
    Find the CSV data directory.
    
    Checks directories in priority order and validates that
    required files exist.
    
    Returns:
        Path to the CSV directory
        
    Raises:
        FileNotFoundError: If no valid directory found
    """
    for candidate in CSV_DIR_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            # Check if required files exist
            has_required = all((candidate / f).exists() for f in REQUIRED_CSV_FILES)
            if has_required:
                logger.debug(f"Found CSV directory: {candidate}")
                return candidate
    
    raise FileNotFoundError(
        f"Could not find CSV data directory with required files.\n"
        f"Checked: {[str(p) for p in CSV_DIR_CANDIDATES]}\n"
        f"Required files: {REQUIRED_CSV_FILES}"
    )
