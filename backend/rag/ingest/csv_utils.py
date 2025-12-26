"""
CSV utility functions for data ingestion.

This module provides utilities for locating and validating
CSV data directories used by the RAG ingestion pipeline.

This is the single source of truth for CSV directory location.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Get backend root directory
_BACKEND_ROOT = Path(__file__).parent.parent.parent

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


def validate_csv_dir(csv_dir: Path) -> tuple[bool, list[str]]:
    """
    Validate that a CSV directory contains all required files.
    
    Args:
        csv_dir: Path to the CSV directory
        
    Returns:
        Tuple of (is_valid, missing_files)
    """
    missing = []
    for filename in REQUIRED_CSV_FILES:
        if not (csv_dir / filename).exists():
            missing.append(filename)
    
    return len(missing) == 0, missing


def get_csv_path(filename: str) -> Optional[Path]:
    """
    Get the full path to a specific CSV file.
    
    Args:
        filename: Name of the CSV file (e.g., "companies.csv")
        
    Returns:
        Full path to the file, or None if not found
    """
    try:
        csv_dir = find_csv_dir()
    except FileNotFoundError:
        return None
    
    file_path = csv_dir / filename
    if file_path.exists():
        return file_path
    
    return None


def list_csv_files(csv_dir: Optional[Path] = None) -> list[str]:
    """
    List all CSV files in the data directory.
    
    Args:
        csv_dir: Optional specific directory to check
        
    Returns:
        List of CSV filenames
    """
    if csv_dir is None:
        try:
            csv_dir = find_csv_dir()
        except FileNotFoundError:
            return []
    
    if not csv_dir.exists():
        return []
    
    return [f.name for f in csv_dir.glob("*.csv")]
