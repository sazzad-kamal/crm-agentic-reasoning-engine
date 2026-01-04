"""
Company operations for CRM Data Store.

Provides company resolution, lookup, and search functionality.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.agent.datastore.base import DataStoreMixinProtocol

    _MixinBase = DataStoreMixinProtocol
else:
    _MixinBase = object


class CompanyMixin(_MixinBase):
    """Mixin providing company-related operations."""

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

        Returns list of company dicts.
        """
        if not partial_name:
            return []

        self._build_company_cache()

        all_names = list(self._company_names_cache.keys())
        matches = get_close_matches(partial_name.lower(), all_names, n=limit, cutoff=0.4)

        results = []
        for name in matches:
            company_id = self._company_names_cache[name]
            company = self.get_company(company_id)
            if company:
                results.append(company)

        return results

    def get_company(self, company_id: str) -> dict | None:
        """Get company details by ID."""
        self._ensure_table("companies")
        return self._fetch_one_dict("SELECT * FROM companies WHERE company_id = ?", [company_id])

    def search_companies(
        self,
        query: str = "",
        industry: str = "",
        segment: str = "",
        status: str = "",
        region: str = "",
        limit: int = 20,
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
            f"SELECT * FROM companies WHERE {where_clause} ORDER BY name LIMIT {limit}", params
        )
