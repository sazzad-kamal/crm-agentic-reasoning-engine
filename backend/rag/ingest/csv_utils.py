"""
CSV utility functions for data ingestion.

This module provides utilities for locating and validating
CSV data directories used by the RAG ingestion pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Candidate paths for CSV directory (in order of preference)
CSV_DIR_CANDIDATES = [
    Path(__file__).parent.parent / "data" / "csv",
    Path(__file__).parent.parent.parent / "data" / "csv",
    Path.cwd() / "backend" / "data" / "csv",
    Path.cwd() / "data" / "csv",
]

# Required files for private text building
REQUIRED_CSV_FILES = [
    "history.csv",
    "opportunities.csv",
    "attachments.csv",
    "companies.csv",
    "contacts.csv",
]


def find_csv_dir() -> Optional[Path]:
    """
    Find the CSV data directory by checking candidate paths.
    
    Returns:
        Path to the CSV directory, or None if not found
    """
    for candidate in CSV_DIR_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            # Check if at least one CSV file exists
            csv_files = list(candidate.glob("*.csv"))
            if csv_files:
                logger.debug(f"Found CSV directory: {candidate}")
                return candidate
    
    logger.warning("Could not find CSV data directory")
    return None


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
    csv_dir = find_csv_dir()
    if csv_dir is None:
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
        csv_dir = find_csv_dir()
    
    if csv_dir is None or not csv_dir.exists():
        return []
    
    return [f.name for f in csv_dir.glob("*.csv")]
