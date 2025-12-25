"""
Unified company resolution utility.

Consolidates company name/ID resolution logic previously duplicated across:
- backend/rag/pipeline/account.py
- backend/rag/pipeline/company.py
- backend/agent/datastore.py

Provides a single, consistent API for resolving company names to IDs.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional
from difflib import get_close_matches

import pandas as pd


logger = logging.getLogger(__name__)


# =============================================================================
# CSV Path Resolution
# =============================================================================

def _find_csv_dir() -> Optional[Path]:
    """Find the CSV data directory."""
    # Check from backend/common/
    candidates = [
        Path(__file__).parent.parent / "data" / "csv",
        Path(__file__).parent.parent / "data" / "crm",
        Path.cwd() / "backend" / "data" / "csv",
        Path.cwd() / "backend" / "data" / "crm",
    ]
    for p in candidates:
        if p.exists() and (p / "companies.csv").exists():
            return p
    return None


@lru_cache(maxsize=1)
def _load_companies_df() -> pd.DataFrame:
    """
    Load companies DataFrame (cached).
    
    Returns:
        DataFrame with company data
        
    Raises:
        FileNotFoundError: If companies.csv cannot be found
    """
    csv_dir = _find_csv_dir()
    if csv_dir is None:
        raise FileNotFoundError("Could not find CSV directory with companies.csv")
    
    companies_path = csv_dir / "companies.csv"
    return pd.read_csv(companies_path)


def clear_cache():
    """Clear the companies cache (useful for testing)."""
    _load_companies_df.cache_clear()


# =============================================================================
# Company Resolver
# =============================================================================

class CompanyResolver:
    """
    Unified company resolution with multiple matching strategies.
    
    Provides:
    - Exact ID matching
    - Exact name matching (case-insensitive)
    - Prefix matching
    - Contains matching
    - Fuzzy matching (using difflib)
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize resolver.
        
        Args:
            df: Optional DataFrame with company data. If not provided,
                loads from CSV automatically.
        """
        self._df = df
        self._name_to_id: dict[str, str] = {}
        self._id_set: set[str] = set()
        self._initialized = False
    
    @property
    def df(self) -> pd.DataFrame:
        """Get the companies DataFrame (lazy loaded)."""
        if self._df is None:
            self._df = _load_companies_df()
        return self._df
    
    def _ensure_initialized(self):
        """Build internal caches."""
        if self._initialized:
            return
        
        df = self.df
        self._name_to_id = {
            str(row.get("name", "")).strip().lower(): str(row.get("company_id", ""))
            for _, row in df.iterrows()
        }
        self._id_set = {str(row.get("company_id", "")) for _, row in df.iterrows()}
        self._initialized = True
    
    def resolve(self, name_or_id: str) -> Optional[str]:
        """
        Resolve a company name or ID to a company_id.
        
        Tries in order:
        1. Exact ID match
        2. Exact name match (case-insensitive)
        3. Prefix match
        4. Contains match
        5. Fuzzy match (cutoff=0.6)
        
        Args:
            name_or_id: Company ID or name to resolve
            
        Returns:
            The company_id if found, None otherwise
        """
        if not name_or_id:
            return None
        
        self._ensure_initialized()
        name_or_id = str(name_or_id).strip()
        
        # 1. Exact ID match
        if name_or_id in self._id_set:
            return name_or_id
        
        # 2. Exact name match
        lower_name = name_or_id.lower()
        if lower_name in self._name_to_id:
            return self._name_to_id[lower_name]
        
        # 3. Prefix match
        for name, company_id in self._name_to_id.items():
            if name.startswith(lower_name) or lower_name.startswith(name):
                return company_id
        
        # 4. Contains match
        for name, company_id in self._name_to_id.items():
            if lower_name in name or name in lower_name:
                return company_id
        
        # 5. Fuzzy match
        all_names = list(self._name_to_id.keys())
        matches = get_close_matches(lower_name, all_names, n=1, cutoff=0.6)
        if matches:
            return self._name_to_id[matches[0]]
        
        logger.debug(f"Could not resolve company: {name_or_id}")
        return None
    
    def get_name(self, company_id: str) -> Optional[str]:
        """
        Get company name from company_id.
        
        Args:
            company_id: The company ID to look up
            
        Returns:
            The company name if found, None otherwise
        """
        if not company_id:
            return None
        
        df = self.df
        match = df[df["company_id"] == company_id]
        if not match.empty:
            return str(match.iloc[0]["name"])
        return None
    
    def get_close_matches(
        self,
        partial_name: str,
        limit: int = 5,
        cutoff: float = 0.4,
    ) -> list[dict]:
        """
        Get companies matching a partial name (fuzzy match).
        
        Args:
            partial_name: Partial name to search for
            limit: Maximum number of results
            cutoff: Similarity cutoff (0-1)
        
        Returns:
            List of {company_id, name} dicts
        """
        if not partial_name:
            return []
        
        self._ensure_initialized()
        
        all_names = list(self._name_to_id.keys())
        matches = get_close_matches(
            partial_name.lower(), all_names, n=limit, cutoff=cutoff
        )
        
        results = []
        for name in matches:
            company_id = self._name_to_id[name]
            # Get full company record
            df = self.df
            row = df[df["company_id"] == company_id]
            if not row.empty:
                results.append(row.iloc[0].to_dict())
        
        return results
    
    def validate_id(self, company_id: str) -> bool:
        """Check if a company ID exists."""
        self._ensure_initialized()
        return company_id in self._id_set
    
    def list_all(self) -> list[dict]:
        """Get all companies as list of dicts."""
        return self.df.to_dict("records")


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_resolver: Optional[CompanyResolver] = None


def get_resolver() -> CompanyResolver:
    """Get the default company resolver instance."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = CompanyResolver()
    return _default_resolver


def resolve_company_id(name_or_id: str) -> Optional[str]:
    """
    Resolve a company name or ID to a company_id.
    
    Convenience function using the default resolver.
    """
    return get_resolver().resolve(name_or_id)


def get_company_name(company_id: str) -> Optional[str]:
    """
    Get company name from company_id.
    
    Convenience function using the default resolver.
    """
    return get_resolver().get_name(company_id)


def get_company_matches(partial_name: str, limit: int = 5) -> list[dict]:
    """
    Get companies matching a partial name.
    
    Convenience function using the default resolver.
    """
    return get_resolver().get_close_matches(partial_name, limit=limit)


def load_companies_df() -> pd.DataFrame:
    """
    Load companies DataFrame.
    
    For backwards compatibility with existing code.
    """
    return _load_companies_df()
