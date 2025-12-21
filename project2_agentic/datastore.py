"""
CRM Data Store using DuckDB.

Provides lazy loading and query methods for CRM CSV data.
Uses DuckDB's read_csv_auto() for efficient data access.
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from difflib import get_close_matches
import duckdb


# =============================================================================
# Configuration
# =============================================================================

# CSV table files
CSV_TABLES = {
    "companies": "companies.csv",
    "contacts": "contacts.csv",
    "activities": "activities.csv",
    "history": "history.csv",
    "opportunities": "opportunities.csv",
    "groups": "groups.csv",
    "group_members": "group_members.csv",
    "attachments": "attachments.csv",
    "opportunity_descriptions": "opportunity_descriptions.csv",
}

OPTIONAL_TABLES = {"groups", "group_members", "attachments", "opportunity_descriptions"}
REQUIRED_TABLES = {"companies", "contacts", "activities", "history", "opportunities"}


def get_csv_base_path() -> Path:
    """
    Get the base path for CSV files with fallback logic.
    
    Priority:
    1. data/crm/ (if exists)
    2. data/csv/ (fallback)
    3. Raise error if neither exists
    """
    # Get project root (go up from this file's directory)
    project_root = Path(__file__).parent.parent
    
    # Check preferred path
    preferred = project_root / "data" / "crm"
    if preferred.exists() and preferred.is_dir():
        return preferred
    
    # Check fallback path
    fallback = project_root / "data" / "csv"
    if fallback.exists() and fallback.is_dir():
        return fallback
    
    raise FileNotFoundError(
        f"Could not find CSV data directory. "
        f"Checked: {preferred} and {fallback}. "
        f"Please ensure one of these directories exists with CRM CSV files."
    )


# =============================================================================
# CRM Data Store
# =============================================================================

class CRMDataStore:
    """
    DuckDB-based CRM data store with lazy loading.
    
    Loads CSV files on first access and provides query methods
    for common CRM operations.
    """
    
    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize the data store.
        
        Args:
            csv_path: Optional path to CSV directory. 
                      If not provided, uses auto-detection.
        """
        self._csv_path = csv_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._loaded_tables: set[str] = set()
        self._company_names_cache: Optional[dict[str, str]] = None  # name -> id
        self._company_ids_cache: Optional[set[str]] = None
    
    @property
    def csv_path(self) -> Path:
        """Get the CSV base path (lazy resolved)."""
        if self._csv_path is None:
            self._csv_path = get_csv_base_path()
        return self._csv_path
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get the DuckDB connection (lazy created)."""
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
        return self._conn
    
    def _ensure_table(self, table_name: str) -> bool:
        """
        Ensure a table is loaded into DuckDB.
        
        Returns True if table is available, False if not found.
        """
        if table_name in self._loaded_tables:
            return True
        
        filename = CSV_TABLES.get(table_name)
        if not filename:
            return False
        
        csv_file = self.csv_path / filename
        if not csv_file.exists():
            if table_name in REQUIRED_TABLES:
                raise FileNotFoundError(f"Required CSV file not found: {csv_file}")
            return False
        
        # Load using DuckDB's read_csv_auto
        try:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} AS 
                SELECT * FROM read_csv_auto('{csv_file.as_posix()}')
            """)
            self._loaded_tables.add(table_name)
            return True
        except Exception as e:
            print(f"Warning: Failed to load {table_name}: {e}")
            return False
    
    def _ensure_core_tables(self):
        """Load all required core tables."""
        for table in REQUIRED_TABLES:
            self._ensure_table(table)
    
    def _build_company_cache(self):
        """Build the company name -> ID cache."""
        if self._company_names_cache is not None:
            return
        
        self._ensure_table("companies")
        
        result = self.conn.execute(
            "SELECT company_id, name FROM companies"
        ).fetchall()
        
        self._company_names_cache = {}
        self._company_ids_cache = set()
        
        for company_id, name in result:
            self._company_ids_cache.add(company_id)
            # Store lowercase name for matching
            self._company_names_cache[name.lower()] = company_id
    
    def _safe_parse_date(self, value) -> Optional[datetime]:
        """Safely parse a date/datetime value."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            # Try ISO format first
            if "T" in str(value):
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None
    
    def _get_date_cutoff(self, days: int) -> str:
        """Get ISO date string for N days ago."""
        cutoff = datetime.now() - timedelta(days=days)
        return cutoff.isoformat()
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def resolve_company_id(self, name_or_id: str) -> Optional[str]:
        """
        Resolve a company name or ID to a company_id.
        
        Args:
            name_or_id: Either a company_id or a company name
            
        Returns:
            The company_id if found, else None
        """
        if not name_or_id:
            return None
        
        self._build_company_cache()
        
        # Check exact ID match
        if name_or_id in self._company_ids_cache:
            return name_or_id
        
        # Check exact name match (case-insensitive)
        lower_name = name_or_id.lower()
        if lower_name in self._company_names_cache:
            return self._company_names_cache[lower_name]
        
        # Fuzzy match on company names
        all_names = list(self._company_names_cache.keys())
        matches = get_close_matches(lower_name, all_names, n=1, cutoff=0.6)
        if matches:
            return self._company_names_cache[matches[0]]
        
        return None
    
    def get_company_name_matches(self, partial_name: str, limit: int = 5) -> list[dict]:
        """
        Get companies matching a partial name (fuzzy match).
        
        Returns list of {company_id, name, similarity} dicts.
        """
        if not partial_name:
            return []
        
        self._build_company_cache()
        
        all_names = list(self._company_names_cache.keys())
        matches = get_close_matches(
            partial_name.lower(), all_names, n=limit, cutoff=0.4
        )
        
        results = []
        for name in matches:
            company_id = self._company_names_cache[name]
            company = self.get_company(company_id)
            if company:
                results.append(company)
        
        return results
    
    def get_company(self, company_id: str) -> Optional[dict]:
        """
        Get company details by ID.
        
        Returns dict with company fields, or None if not found.
        """
        self._ensure_table("companies")
        
        result = self.conn.execute(
            "SELECT * FROM companies WHERE company_id = ?",
            [company_id]
        ).fetchone()
        
        if not result:
            return None
        
        # Get column names
        columns = [desc[0] for desc in self.conn.description]
        return dict(zip(columns, result))
    
    def get_all_companies(self) -> list[dict]:
        """Get all companies."""
        self._ensure_table("companies")
        
        result = self.conn.execute("SELECT * FROM companies").fetchall()
        columns = [desc[0] for desc in self.conn.description]
        
        return [dict(zip(columns, row)) for row in result]
    
    def get_recent_activities(
        self,
        company_id: str,
        days: int = 90,
        limit: int = 20
    ) -> list[dict]:
        """
        Get recent activities for a company.
        
        Args:
            company_id: The company ID
            days: Number of days to look back
            limit: Maximum number of activities
            
        Returns:
            List of activity dicts sorted by date (newest first)
        """
        self._ensure_table("activities")
        
        cutoff = self._get_date_cutoff(days)
        
        # Query with date filtering
        # Note: due_datetime may have timezone info, we do substring comparison
        try:
            result = self.conn.execute(f"""
                SELECT * FROM activities 
                WHERE company_id = ?
                AND (
                    due_datetime >= '{cutoff}' 
                    OR created_at >= '{cutoff}'
                    OR due_datetime IS NULL
                )
                ORDER BY COALESCE(due_datetime, created_at) DESC
                LIMIT {limit}
            """, [company_id]).fetchall()
        except Exception:
            # Fallback without date filter if date parsing fails
            result = self.conn.execute(f"""
                SELECT * FROM activities 
                WHERE company_id = ?
                ORDER BY created_at DESC
                LIMIT {limit}
            """, [company_id]).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def get_recent_history(
        self,
        company_id: str,
        days: int = 90,
        limit: int = 20
    ) -> list[dict]:
        """
        Get recent history entries for a company.
        
        Args:
            company_id: The company ID
            days: Number of days to look back
            limit: Maximum number of entries
            
        Returns:
            List of history dicts sorted by date (newest first)
        """
        self._ensure_table("history")
        
        cutoff = self._get_date_cutoff(days)
        
        try:
            result = self.conn.execute(f"""
                SELECT * FROM history 
                WHERE company_id = ?
                AND occurred_at >= '{cutoff}'
                ORDER BY occurred_at DESC
                LIMIT {limit}
            """, [company_id]).fetchall()
        except Exception:
            # Fallback without date filter
            result = self.conn.execute(f"""
                SELECT * FROM history 
                WHERE company_id = ?
                ORDER BY occurred_at DESC
                LIMIT {limit}
            """, [company_id]).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def get_open_opportunities(
        self,
        company_id: str,
        limit: int = 20
    ) -> list[dict]:
        """
        Get open opportunities for a company.
        
        Filters out closed stages (Closed Won, Closed Lost).
        """
        self._ensure_table("opportunities")
        
        result = self.conn.execute(f"""
            SELECT * FROM opportunities 
            WHERE company_id = ?
            AND LOWER(stage) NOT LIKE '%closed%'
            ORDER BY value DESC
            LIMIT {limit}
        """, [company_id]).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def get_pipeline_summary(self, company_id: str) -> dict:
        """
        Get pipeline summary for a company.
        
        Returns:
            Dict with:
            - stages: {stage_name: {count, total_value}}
            - total_count: int
            - total_value: float
        """
        self._ensure_table("opportunities")
        
        result = self.conn.execute("""
            SELECT 
                stage, 
                COUNT(*) as count,
                SUM(COALESCE(value, 0)) as total_value
            FROM opportunities 
            WHERE company_id = ?
            AND LOWER(stage) NOT LIKE '%closed%'
            GROUP BY stage
        """, [company_id]).fetchall()
        
        stages = {}
        total_count = 0
        total_value = 0.0
        
        for stage, count, value in result:
            stages[stage] = {"count": count, "total_value": float(value or 0)}
            total_count += count
            total_value += float(value or 0)
        
        return {
            "stages": stages,
            "total_count": total_count,
            "total_value": total_value,
        }
    
    def get_upcoming_renewals(
        self,
        days: int = 90,
        limit: int = 20
    ) -> list[dict]:
        """
        Get companies with upcoming renewals.
        
        Args:
            days: Number of days to look ahead
            limit: Maximum number of results
            
        Returns:
            List of company dicts with renewal_date within the window
        """
        self._ensure_table("companies")
        
        today = datetime.now().strftime("%Y-%m-%d")
        cutoff = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        
        try:
            result = self.conn.execute(f"""
                SELECT * FROM companies 
                WHERE renewal_date >= '{today}'
                AND renewal_date <= '{cutoff}'
                AND status = 'Active'
                ORDER BY renewal_date ASC
                LIMIT {limit}
            """).fetchall()
        except Exception:
            # Fallback: just get all with renewal_date
            result = self.conn.execute(f"""
                SELECT * FROM companies 
                WHERE renewal_date IS NOT NULL
                AND status = 'Active'
                ORDER BY renewal_date ASC
                LIMIT {limit}
            """).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def get_contacts_for_company(
        self,
        company_id: str,
        limit: int = 10
    ) -> list[dict]:
        """Get contacts for a company."""
        self._ensure_table("contacts")
        
        result = self.conn.execute(f"""
            SELECT * FROM contacts 
            WHERE company_id = ?
            LIMIT {limit}
        """, [company_id]).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def execute_query(self, query: str, params: list = None) -> list[dict]:
        """
        Execute a raw SQL query (for advanced use cases).
        
        Returns list of dicts.
        """
        self._ensure_core_tables()
        
        if params:
            result = self.conn.execute(query, params).fetchall()
        else:
            result = self.conn.execute(query).fetchall()
        
        if not result:
            return []
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]


# =============================================================================
# Singleton instance
# =============================================================================

_datastore: Optional[CRMDataStore] = None


def get_datastore() -> CRMDataStore:
    """Get the singleton CRMDataStore instance."""
    global _datastore
    if _datastore is None:
        _datastore = CRMDataStore()
    return _datastore


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing CRMDataStore")
    print("=" * 60)
    
    ds = get_datastore()
    print(f"CSV path: {ds.csv_path}")
    
    # Test company resolution
    print("\nResolving 'Acme Manufacturing'...")
    cid = ds.resolve_company_id("Acme Manufacturing")
    print(f"  -> {cid}")
    
    print("\nResolving 'ACME-MFG'...")
    cid = ds.resolve_company_id("ACME-MFG")
    print(f"  -> {cid}")
    
    print("\nResolving 'acme' (fuzzy)...")
    cid = ds.resolve_company_id("acme")
    print(f"  -> {cid}")
    
    # Get company
    if cid:
        print(f"\nCompany info for {cid}:")
        company = ds.get_company(cid)
        for k, v in (company or {}).items():
            print(f"  {k}: {v}")
    
    # Get recent activities
    if cid:
        print(f"\nRecent activities for {cid}:")
        activities = ds.get_recent_activities(cid, days=365)
        for act in activities[:3]:
            print(f"  - {act.get('type')}: {act.get('subject')}")
    
    # Get pipeline
    if cid:
        print(f"\nPipeline summary for {cid}:")
        pipeline = ds.get_pipeline_summary(cid)
        print(f"  Total open: {pipeline['total_count']} deals, ${pipeline['total_value']}")
    
    # Get renewals
    print("\nUpcoming renewals (90 days):")
    renewals = ds.get_upcoming_renewals(days=90)
    for r in renewals[:3]:
        print(f"  - {r.get('name')} ({r.get('company_id')}): {r.get('renewal_date')}")
