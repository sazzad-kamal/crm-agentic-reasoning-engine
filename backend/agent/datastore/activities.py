"""
Activity and history operations for CRM Data Store.

Provides activity queries, history lookup, and search functionality.
"""


class ActivityMixin:
    """Mixin providing activity and history operations."""

    def get_recent_activities(self, company_id: str, days: int = 90, limit: int = 20) -> list[dict]:
        """Get recent activities for a company, sorted by date (newest first)."""
        self._ensure_table("activities")

        cutoff = self._get_date_cutoff(days)

        try:
            result = self.conn.execute(
                f"""
                SELECT * FROM activities
                WHERE company_id = ?
                AND (
                    due_datetime >= '{cutoff}'
                    OR created_at >= '{cutoff}'
                    OR due_datetime IS NULL
                )
                ORDER BY COALESCE(due_datetime, created_at) DESC
                LIMIT {limit}
            """,
                [company_id],
            ).fetchall()
        except Exception:
            result = self.conn.execute(
                f"""
                SELECT * FROM activities
                WHERE company_id = ?
                ORDER BY created_at DESC
                LIMIT {limit}
            """,
                [company_id],
            ).fetchall()

        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def get_recent_history(self, company_id: str, days: int = 90, limit: int = 20) -> list[dict]:
        """Get recent history entries for a company, sorted by date (newest first)."""
        self._ensure_table("history")

        cutoff = self._get_date_cutoff(days)

        try:
            result = self.conn.execute(
                f"""
                SELECT * FROM history
                WHERE company_id = ?
                AND occurred_at >= '{cutoff}'
                ORDER BY occurred_at DESC
                LIMIT {limit}
            """,
                [company_id],
            ).fetchall()
        except Exception:
            result = self.conn.execute(
                f"""
                SELECT * FROM history
                WHERE company_id = ?
                ORDER BY occurred_at DESC
                LIMIT {limit}
            """,
                [company_id],
            ).fetchall()

        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]

    def search_activities(
        self, activity_type: str = "", days: int = 30, company_id: str = "", limit: int = 30
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
            params,
        )
