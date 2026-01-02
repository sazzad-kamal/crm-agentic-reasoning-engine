"""
Contact operations for CRM Data Store.

Provides contact lookup and search functionality.
"""


class ContactMixin:
    """Mixin providing contact-related operations."""

    def get_contacts_for_company(self, company_id: str, limit: int = 10) -> list[dict]:
        """Get contacts for a company."""
        self._ensure_table("contacts")
        return self._fetch_all_dicts(
            f"SELECT * FROM contacts WHERE company_id = ? LIMIT {limit}", [company_id]
        )

    def get_contact(self, contact_id: str) -> dict | None:
        """Get contact by ID."""
        self._ensure_table("contacts")
        return self._fetch_one_dict("SELECT * FROM contacts WHERE contact_id = ?", [contact_id])

    def search_contacts(
        self,
        query: str = "",
        role: str = "",
        job_title: str = "",
        company_id: str = "",
        limit: int = 20,
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
            f"SELECT * FROM contacts WHERE {where_clause} LIMIT {limit}", params
        )
