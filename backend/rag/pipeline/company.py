"""
Company resolution utilities for account-scoped pipelines.

This module handles company data loading and resolution from company names
using the companies CSV data.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _find_csv_dir() -> Optional[Path]:
    """Find the CSV data directory."""
    candidates = [
        Path(__file__).parent.parent.parent / "data" / "csv",
        Path(__file__).parent.parent.parent.parent / "data" / "csv",
        Path.cwd() / "backend" / "data" / "csv",
    ]
    for p in candidates:
        if p.exists() and (p / "companies.csv").exists():
            return p
    return None


def load_companies_df() -> pd.DataFrame:
    """
    Load companies DataFrame.
    
    Returns:
        DataFrame with company data
        
    Raises:
        FileNotFoundError: If companies.csv cannot be found
    """
    csv_dir = _find_csv_dir()
    if csv_dir is None:
        raise FileNotFoundError("Could not find CSV directory with companies.csv")
    
    companies_path = csv_dir / "companies.csv"
    if not companies_path.exists():
        raise FileNotFoundError(f"companies.csv not found in {csv_dir}")
    
    return pd.read_csv(companies_path)


def resolve_company_id(company_name: str) -> Optional[str]:
    """
    Resolve a company name to a company_id using fuzzy matching.
    
    Args:
        company_name: The company name to resolve
        
    Returns:
        The company_id if found, None otherwise
    """
    df = load_companies_df()
    if df is None or df.empty:
        return None
    
    # Normalize input
    normalized = company_name.strip().lower()
    if not normalized:
        return None
    
    # Try exact match first
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip().lower()
        if name == normalized:
            return str(row.get("company_id", ""))
    
    # Try prefix match
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip().lower()
        if name.startswith(normalized) or normalized.startswith(name):
            return str(row.get("company_id", ""))
    
    # Try contains match
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip().lower()
        if normalized in name or name in normalized:
            return str(row.get("company_id", ""))
    
    logger.debug(f"Could not resolve company name: {company_name}")
    return None


def get_company_name(company_id: str) -> Optional[str]:
    """
    Get company name from company_id.
    
    Args:
        company_id: The company ID to look up
        
    Returns:
        The company name if found, None otherwise
    """
    df = load_companies_df()
    if df is None or df.empty:
        return None
    
    for _, row in df.iterrows():
        if str(row.get("company_id", "")) == company_id:
            return str(row.get("name", ""))
    
    return None
