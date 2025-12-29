"""
CRM Data Store using DuckDB.

Provides lazy loading and query methods for CRM CSV data.
Uses DuckDB's read_csv_auto() for efficient data access.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from difflib import get_close_matches
from typing import Any
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

REQUIRED_TABLES = {"companies", "contacts", "activities", "history", "opportunities"}


def get_csv_base_path() -> Path:
    """
    Get the base path for CSV files with fallback logic.
    
    Priority:
    1. data/crm/ (if exists)
    2. data/csv/ (fallback)
    3. Raise error if neither exists
    """
    # Get backend root (go up from backend/agent/ to backend/)
    backend_root = Path(__file__).parent.parent
    
    # Check preferred path
    preferred = backend_root / "data" / "crm"
    if preferred.exists() and preferred.is_dir():
        return preferred
    
    # Check fallback path
    fallback = backend_root / "data" / "csv"
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

    Supports context manager for automatic cleanup:
        with CRMDataStore() as store:
            data = store.get_company("ACME-MFG")
    """

    def __init__(self, csv_path: Path | None = None) -> None:
        """
        Initialize the data store.

        Args:
            csv_path: Optional path to CSV directory.
                      If not provided, uses auto-detection.
        """
        self._csv_path = csv_path
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._loaded_tables: set[str] = set()
        self._company_names_cache: dict[str, str] | None = None  # name -> id
        self._company_ids_cache: set[str] | None = None

    def __enter__(self) -> "CRMDataStore":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager and cleanup resources."""
        self.close()
        return False

    def close(self) -> None:
        """Close the database connection and cleanup resources."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self._conn = None
        self._loaded_tables.clear()
        self._company_names_cache = None
        self._company_ids_cache = None
    
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
    
    def _fetch_one_dict(self, query: str, params: list[Any] | None = None) -> dict[str, Any] | None:
        """Execute query and return first row as dict, or None."""
        result = self.conn.execute(query, params or []).fetchone()
        if not result:
            return None
        return dict(zip([d[0] for d in self.conn.description], result))

    def _fetch_all_dicts(self, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        """Execute query and return all rows as list of dicts."""
        result = self.conn.execute(query, params or []).fetchall()
        if not result:
            return []
        columns = [d[0] for d in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def _ensure_core_tables(self) -> None:
        """Load all required core tables."""
        for table in REQUIRED_TABLES:
            self._ensure_table(table)

    def _build_company_cache(self) -> None:
        """Build the company name -> ID cache."""
        if self._company_names_cache is not None:
            return
        
        self._ensure_table("companies")
        
        result = self.conn.execute(
            "SELECT company_id, name FROM companies"
        ).fetchall()
        
        self._company_names_cache = {name.lower(): cid for cid, name in result}
        self._company_ids_cache = {cid for cid, _ in result}
    
    def _get_date_cutoff(self, days: int) -> str:
        """Get ISO date string for N days ago."""
        cutoff = datetime.now() - timedelta(days=days)
        return cutoff.isoformat()
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def resolve_company_id(self, name_or_id: str) -> str | None:
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
    
    def get_company(self, company_id: str) -> dict | None:
        """
        Get company details by ID.
        
        Returns dict with company fields, or None if not found.
        """
        self._ensure_table("companies")
        return self._fetch_one_dict(
            "SELECT * FROM companies WHERE company_id = ?",
            [company_id]
        )
    
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
        
        stages = {stage: {"count": count, "total_value": float(value or 0)}
                  for stage, count, value in result}
        return {
            "stages": stages,
            "total_count": sum(s["count"] for s in stages.values()),
            "total_value": sum(s["total_value"] for s in stages.values()),
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
        return self._fetch_all_dicts(
            f"SELECT * FROM contacts WHERE company_id = ? LIMIT {limit}",
            [company_id]
        )

    def get_contact(self, contact_id: str) -> dict | None:
        """Get contact by ID."""
        self._ensure_table("contacts")
        return self._fetch_one_dict(
            "SELECT * FROM contacts WHERE contact_id = ?",
            [contact_id]
        )

    def search_contacts(
        self,
        query: str = "",
        role: str = "",
        job_title: str = "",
        company_id: str = "",
        limit: int = 20
    ) -> list[dict]:
        """
        Search contacts by name, role, job_title, or company.
        
        Args:
            query: Search term for name/email
            role: Filter by role (e.g., "Decision Maker")
            job_title: Filter by job title (partial match)
            company_id: Filter by company
            limit: Max results
        """
        self._ensure_table("contacts")
        
        conditions = []
        params = []
        
        if query:
            conditions.append("""
                (LOWER(first_name) LIKE ? 
                OR LOWER(last_name) LIKE ? 
                OR LOWER(email) LIKE ?
                OR LOWER(first_name || ' ' || last_name) LIKE ?)
            """)
            q = f"%{query.lower()}%"
            params.extend([q, q, q, q])
        
        if role:
            conditions.append("LOWER(role) LIKE ?")
            params.append(f"%{role.lower()}%")
        
        if job_title:
            conditions.append("LOWER(job_title) LIKE ?")
            params.append(f"%{job_title.lower()}%")
        
        if company_id:
            conditions.append("company_id = ?")
            params.append(company_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        return self._fetch_all_dicts(
            f"SELECT * FROM contacts WHERE {where_clause} LIMIT {limit}",
            params
        )

    def search_companies(
        self,
        query: str = "",
        industry: str = "",
        segment: str = "",
        status: str = "",
        region: str = "",
        limit: int = 20
    ) -> list[dict]:
        """
        Search companies by various criteria.
        
        Args:
            query: Search term for name
            industry: Filter by industry
            segment: Filter by segment (SMB, Mid-market, Enterprise)
            status: Filter by status (Active, Churned, etc.)
            region: Filter by region
            limit: Max results
        """
        self._ensure_table("companies")
        
        conditions = []
        params = []
        
        if query:
            conditions.append("LOWER(name) LIKE ?")
            params.append(f"%{query.lower()}%")
        
        if industry:
            conditions.append("LOWER(industry) LIKE ?")
            params.append(f"%{industry.lower()}%")
        
        if segment:
            conditions.append("LOWER(segment) LIKE ?")
            params.append(f"%{segment.lower()}%")
        
        if status:
            conditions.append("LOWER(status) LIKE ?")
            params.append(f"%{status.lower()}%")
        
        if region:
            conditions.append("LOWER(region) LIKE ?")
            params.append(f"%{region.lower()}%")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        return self._fetch_all_dicts(
            f"SELECT * FROM companies WHERE {where_clause} ORDER BY name LIMIT {limit}",
            params
        )

    def get_group(self, group_id: str) -> dict | None:
        """Get group by ID."""
        if not self._ensure_table("groups"):
            return None
        return self._fetch_one_dict(
            "SELECT * FROM groups WHERE group_id = ?",
            [group_id]
        )

    def get_all_groups(self) -> list[dict]:
        """Get all groups."""
        if not self._ensure_table("groups"):
            return []
        return self._fetch_all_dicts("SELECT * FROM groups ORDER BY name")

    def get_group_members(self, group_id: str, limit: int = 50) -> list[dict]:
        """
        Get companies in a group.
        
        Returns list of company dicts for companies in the group.
        """
        if not self._ensure_table("group_members"):
            return []
        self._ensure_table("companies")
        
        return self._fetch_all_dicts(f"""
            SELECT c.* FROM companies c
            INNER JOIN group_members gm ON c.company_id = gm.company_id
            WHERE gm.group_id = ?
            LIMIT {limit}
        """, [group_id])

    def search_attachments(
        self,
        query: str = "",
        company_id: str = "",
        file_type: str = "",
        limit: int = 20
    ) -> list[dict]:
        """
        Search attachments by title, company, or file type.
        """
        if not self._ensure_table("attachments"):
            return []
        
        conditions = []
        params = []
        
        if query:
            conditions.append("(LOWER(title) LIKE ? OR LOWER(summary) LIKE ?)")
            q = f"%{query.lower()}%"
            params.extend([q, q])
        
        if company_id:
            conditions.append("company_id = ?")
            params.append(company_id)
        
        if file_type:
            conditions.append("LOWER(file_type) LIKE ?")
            params.append(f"%{file_type.lower()}%")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        return self._fetch_all_dicts(
            f"SELECT * FROM attachments WHERE {where_clause} ORDER BY created_at DESC LIMIT {limit}",
            params
        )

    def get_all_pipeline_summary(self) -> dict:
        """
        Get pipeline summary across ALL companies.
        
        Returns:
            Dict with total pipeline stats by stage
        """
        self._ensure_table("opportunities")
        
        # Overall stats
        overall = self.conn.execute("""
            SELECT 
                COUNT(*) as total_count,
                SUM(COALESCE(value, 0)) as total_value
            FROM opportunities 
            WHERE LOWER(stage) NOT LIKE '%closed%'
        """).fetchone()
        
        # By stage
        by_stage = self.conn.execute("""
            SELECT 
                stage,
                COUNT(*) as count,
                SUM(COALESCE(value, 0)) as total_value
            FROM opportunities 
            WHERE LOWER(stage) NOT LIKE '%closed%'
            GROUP BY stage
            ORDER BY total_value DESC
        """).fetchall()
        
        # By company (top 10)
        by_company = self.conn.execute("""
            SELECT 
                company_id,
                COUNT(*) as count,
                SUM(COALESCE(value, 0)) as total_value
            FROM opportunities 
            WHERE LOWER(stage) NOT LIKE '%closed%'
            GROUP BY company_id
            ORDER BY total_value DESC
            LIMIT 10
        """).fetchall()
        
        return {
            "total_count": overall[0] if overall else 0,
            "total_value": float(overall[1] or 0) if overall else 0,
            "by_stage": {stage: {"count": count, "value": float(value or 0)} 
                        for stage, count, value in by_stage},
            "top_companies": [{"company_id": cid, "count": count, "value": float(value or 0)}
                            for cid, count, value in by_company],
        }

    def search_activities(
        self,
        activity_type: str = "",
        days: int = 30,
        company_id: str = "",
        limit: int = 30
    ) -> list[dict]:
        """
        Search activities by type, date range, or company.
        
        Args:
            activity_type: Filter by type (e.g., "Demo", "Meeting")
            days: Look back N days
            company_id: Filter by company
            limit: Max results
        """
        self._ensure_table("activities")
        
        conditions = []
        params = []
        cutoff = self._get_date_cutoff(days)
        
        conditions.append(f"(due_datetime >= '{cutoff}' OR created_at >= '{cutoff}')")
        
        if activity_type:
            conditions.append("LOWER(type) LIKE ?")
            params.append(f"%{activity_type.lower()}%")
        
        if company_id:
            conditions.append("company_id = ?")
            params.append(company_id)
        
        where_clause = " AND ".join(conditions)
        
        return self._fetch_all_dicts(
            f"SELECT * FROM activities WHERE {where_clause} ORDER BY due_datetime DESC LIMIT {limit}",
            params
        )


# =============================================================================
# Thread-local datastore instance
# =============================================================================

import threading

_thread_local = threading.local()


def get_datastore() -> CRMDataStore:
    """Get a thread-local CRMDataStore instance.

    Each thread gets its own DuckDB connection to avoid
    concurrent access issues.
    """
    if not hasattr(_thread_local, 'datastore'):
        _thread_local.datastore = CRMDataStore()
    return _thread_local.datastore


__all__ = ["CRMDataStore", "get_datastore", "get_csv_base_path"]
